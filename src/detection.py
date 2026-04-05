import matplotlib.pyplot as plt
import os 
import cv2
import numpy as np
from tqdm import tqdm
import faiss
import tifffile as tiff
import time
import torch

from src.utils import augment_image, dists2map, plot_ref_images
from src.post_eval import mean_top1p

def run_anomaly_detection(
        model,
        object_name,
        data_root,
        n_ref_samples,
        object_anomalies,
        plots_dir,
        save_examples = False,
        masking = None,
        mask_ref_images = False,
        rotation = False,
        knn_metric = 'L2_normalized',
        knn_neighbors = 1,
        faiss_on_cpu = False,
        seed = 0,
        save_patch_dists = True,
        save_tiffs = False,
        use_gen = True,          # NEW
        gen_dirname = "gen",
        use_cad = True, # NEW
        cad_dirname = "cad", # NEW: folder name under object
        fuse_mode = "concat", # NEW: "concat" | "img_diff_concat" | "sum"
        fuse_alpha = 0.5): # NEW: used if fuse_mode == "sum"
    """
    Main function to evaluate the anomaly detection performance of a given object/product.

    Parameters:
    - model: The backbone model for feature extraction (and, in case of DINOv2, masking).
    - object_name: The name of the object/product to evaluate.
    - data_root: The root directory of the dataset.
    - n_ref_samples: The number of reference samples to use for evaluation (k-shot). Set to -1 for full-shot setting.
    - object_anomalies: The anomaly types for each object/product.
    - plots_dir: The directory to save the example plots.
    - save_examples: Whether to save example images and plots. Default is True.
    - masking: Whether to apply DINOv2 to estimate the foreground mask (and discard background patches).
    - rotation: Whether to augment reference samples with rotation.
    - knn_metric: The metric to use for kNN search. Default is 'L2_normalized' (1 - cosine similarity)
    - knn_neighbors: The number of nearest neighbors to consider. Default is 1.
    - seed: The seed value for deterministic sampling in few-shot setting. Default is 0.
    - save_patch_dists: Whether to save the patch distances. Default is True. Required to eval detection.
    - save_tiffs: Whether to save the anomaly maps as TIFF files. Default is False. Required to eval segmentation.
    """

    assert knn_metric in ["L2", "L2_normalized"]
    
    def _read_rgb(path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _cad_path_train(img_name):
        return f"{data_root}/{object_name}/{cad_dirname}/{img_name}"

    def _cad_path_test(type_anomaly, img_name):
        return f"{data_root}/{object_name}/{cad_dirname}/{img_name}"
        
        
    def _gen_path_train(img_name):
        return f"{data_root}/{object_name}/{gen_dirname}/{img_name}"

    def _gen_path_test(type_anomaly, img_name):
        return f"{data_root}/{object_name}/{gen_dirname}/{img_name}"

    
    def _fuse(F_img, F_cad, F_gen):
        # all are numpy arrays [N_patches, D]
        if fuse_mode == "concat":
            return np.concatenate([F_img, F_cad, F_gen], axis=1)  # [N, 3D]

        if fuse_mode == "img_diff_concat":
            # common options (pick one):
            # A) concat raw + diffs (6D total)
            # return np.concatenate([F_img, F_cad, F_gen, (F_img - F_cad), (F_img - F_gen), (F_cad - F_gen)], axis=1)

            # B) concat only diffs (3D total)
            return np.concatenate([(F_img - F_cad), (F_img - F_gen), (F_cad - F_gen)], axis=1)

        if fuse_mode == "sum":
            # weighted sum of 3 sources (simple equal mixing by default)
            # You can customize weights if you want
            #w_img, w_cad, w_gen = 1/3, 1/3, 1/3 recall 1 Ap-0.35? F1=0.43  Acc=0.28
            w_img, w_cad, w_gen = 1/3, 1/3, 1/10
            return w_img * F_img + w_cad * F_cad + w_gen * F_gen

        raise ValueError(f"Unknown fuse_mode={fuse_mode}")
    
    
    
    #def _fuse(F_img, F_cad):
    #    # both are numpy arrays [N_patches, D]
    #    if fuse_mode == "concat":
    #        return np.concatenate([F_img, F_cad], axis=1)              # [N, 2D]
    #    if fuse_mode == "img_diff_concat":
    #        #return np.concatenate([F_img, (F_img - F_cad)], axis=1)    # [N, 2D]
            
    #        return np.concatenate([(F_img - F_cad)], axis=1)    # [N, 2D]
    #    if fuse_mode == "sum":
    #        return fuse_alpha * F_img + (1.0 - fuse_alpha) * F_cad     # [N, D]
    #    
    #    raise ValueError(f"Unknown fuse_mode={fuse_mode}")



    type_anomalies = object_anomalies[object_name]
    # add 'good' to the anomaly types, if exists...
    good_folder = f"{data_root}/{object_name}/test/good/"
    if os.path.exists(good_folder):
        type_anomalies.append('good')
    else:
        print(f"Warning: no 'good' test folder for {object_name} (expected to be at {good_folder})! Just running inference, no evaluation will be performed.")

    # ensure that each type is only evaluated once
    type_anomalies = list(set(type_anomalies))

    # Extract reference features
    features_ref = []
    images_ref = []
    masks_ref = []
    vis_backgroud = []

    img_ref_folder = f"{data_root}/{object_name}/train/good/"
    if n_ref_samples == -1:
        # full-shot setting
        img_ref_samples = sorted(os.listdir(img_ref_folder))
    else:
        # few-shot setting, pick samples in deterministic fashion according to seed
        """
        img_ref_samples = sorted(os.listdir(img_ref_folder))[seed*n_ref_samples:(seed + 1)*n_ref_samples]
        
        print(seed*n_ref_samples)
        print((seed + 1)*n_ref_samples)
        """
        all_imgs = sorted(os.listdir(img_ref_folder))
        N = len(all_imgs)
        K = n_ref_samples  # shots

        if K >= N:
            img_ref_samples = all_imgs  # just take all
        else:
            indices = np.linspace(0, N - 1, K, dtype=int)
            img_ref_samples = [all_imgs[i] for i in indices]
            print(indices)


    if len(img_ref_samples) < n_ref_samples:
        print(f"Warning: Not enough reference samples for {object_name}! Only {len(img_ref_samples)} samples available.")
    
    with torch.inference_mode():
        # start measuring time (feature extraction/memory bank set up)
        start_time = time.time()
        for img_ref_n in tqdm(img_ref_samples, desc="Building memory bank", leave=False):
            # load reference image...
            img_ref = f"{img_ref_folder}{img_ref_n}"
            image_ref = cv2.cvtColor(cv2.imread(img_ref, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

            # augment reference image (if applicable)...
            if rotation:
                img_augmented = augment_image(image_ref)
            else:
                img_augmented = [image_ref]
            for i in range(len(img_augmented)):
                image_ref = img_augmented[i]
                image_ref_tensor, grid_size1 = model.prepare_image(image_ref)
                features_ref_i = model.extract_features(image_ref_tensor)
                
                # compute background mask and discard background patches
                mask_ref = model.compute_background_mask(
            features_ref_i, grid_size1, threshold=10,
            masking_type=(mask_ref_images and masking)
        )

                if use_cad or use_gen:
            # start from real/pfib features
                    F_img = features_ref_i

            # CAD
                    if use_cad:
                        cad_path = _cad_path_train(img_ref_n)
                        cad_img = _read_rgb(cad_path)
                        cad_tensor, cad_grid = model.prepare_image(cad_img)
                        F_cad = model.extract_features(cad_tensor)

                        if tuple(cad_grid) != tuple(grid_size1) or F_cad.shape[0] != F_img.shape[0]:
                            raise RuntimeError(f"CAD/real grid mismatch for {img_ref_n}: real={grid_size1}, cad={cad_grid}")
                    else:
                        F_cad = np.zeros_like(F_img)  # placeholder if disabled

            # GEN
                    if use_gen:
                        gen_path = _gen_path_train(img_ref_n)
                        gen_img = _read_rgb(gen_path)
                        gen_tensor, gen_grid = model.prepare_image(gen_img)
                        F_gen = model.extract_features(gen_tensor)

                        if tuple(gen_grid) != tuple(grid_size1) or F_gen.shape[0] != F_img.shape[0]:
                            raise RuntimeError(f"GEN/real grid mismatch for {img_ref_n}: real={grid_size1}, gen={gen_grid}")
                    else:
                        F_gen = np.zeros_like(F_img)  # placeholder if disabled

                    fused = _fuse(F_img, F_cad, F_gen)
                    features_ref.append(fused[mask_ref])
                else:
                    features_ref.append(features_ref_i[mask_ref])
                
                """
                # compute background mask and discard background patches
                mask_ref = model.compute_background_mask(features_ref_i, grid_size1, threshold=10, masking_type=(mask_ref_images and masking))
                
                if use_cad:
                    cad_path = _cad_path_train(img_ref_n)
                    cad_img = _read_rgb(cad_path)
                    cad_tensor, cad_grid = model.prepare_image(cad_img)
                    features_cad_i = model.extract_features(cad_tensor)

                    # sanity: patch grid must match
                    if tuple(cad_grid) != tuple(grid_size1) or features_cad_i.shape[0] != features_ref_i.shape[0]:
                        raise RuntimeError(f"CAD/real grid mismatch for {img_ref_n}: real={grid_size1}, cad={cad_grid}")

                    fused = _fuse(features_ref_i, features_cad_i)
                    features_ref.append(fused[mask_ref])
                    #print("Ref real dim:", features_ref_i.shape, "Ref CAD dim:", features_cad_i.shape, "Ref fused dim:", fused.shape)
                else:
                    features_ref.append(features_ref_i[mask_ref])
                """
                
                if save_examples:
                    images_ref.append(image_ref)
                    vis_image_background = model.get_embedding_visualization(features_ref_i, grid_size1, mask_ref)
                    masks_ref.append(mask_ref)
                    vis_backgroud.append(vis_image_background)
                
                
        features_ref = np.concatenate(features_ref, axis=0).astype('float32')

        # print(f"Number of reference patches for {object_name}: {features_ref.shape[0]}")
        if faiss_on_cpu:
            # similariy search on CPU
            knn_index = faiss.IndexFlatL2(features_ref.shape[1])
        else:
            # similariy search on GPU
            res = faiss.StandardGpuResources()
            knn_index = faiss.GpuIndexFlatL2(res, features_ref.shape[1])
            # knn_index = faiss.IndexFlatL2(features_ref.shape[1])
            # knn_index = faiss.index_cpu_to_gpu(res, int(model.device[-1]), knn_index)


        if knn_metric == "L2_normalized":
            faiss.normalize_L2(features_ref)
        knn_index.add(features_ref)

        # end measuring time (for memory bank set up; in seconds, same for all test samples of this object)
        time_memorybank = time.time() - start_time

        # plot some reference samples for inspection
        if save_examples:
            plots_dir_ = f"{plots_dir}/{object_name}/"
            plot_ref_images(images_ref, masks_ref, vis_backgroud, grid_size1, plots_dir_, title = "Reference Images", img_names = img_ref_samples)   
        
        inference_times = {}
        anomaly_scores = {}

        idx = 0
        # Evaluate anomalies for each anomaly type (and "good")
        for type_anomaly in tqdm(type_anomalies, desc = f"processing test samples ({object_name})"):
            data_dir = f"{data_root}/{object_name}/test/{type_anomaly}"
            
            if save_patch_dists or save_tiffs:
                os.makedirs(f"{plots_dir}/anomaly_maps/seed={seed}/{object_name}/test/{type_anomaly}", exist_ok=True)
            
            for idx, img_test_nr in tqdm(enumerate(sorted(os.listdir(data_dir))), desc=f"Evaluating object_name'{type_anomaly}'", leave=False, total=len(os.listdir(data_dir))):
                # start measuring time (inference)
                start_time = time.time()
                image_test_path = f"{data_dir}/{img_test_nr}"

                # Extract test features
                image_test = cv2.cvtColor(cv2.imread(image_test_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                image_tensor2, grid_size2 = model.prepare_image(image_test)
                features2 = model.extract_features(image_tensor2)

                # Compute background mask
                if masking:
                    mask2 = model.compute_background_mask(features2, grid_size2, threshold=10, masking_type=masking)
                else:
                    mask2 = np.ones(features2.shape[0], dtype=bool)
        
                """
                # fuse CAD features
                if use_cad:
                    cad_path = _cad_path_test(type_anomaly, img_test_nr)
                    cad_img = _read_rgb(cad_path)
                    cad_tensor2, cad_grid2 = model.prepare_image(cad_img)
                    features2_cad = model.extract_features(cad_tensor2)

                    if tuple(cad_grid2) != tuple(grid_size2) or features2_cad.shape[0] != features2.shape[0]:
                        raise RuntimeError(f"CAD/real grid mismatch for {img_test_nr}: real={grid_size2}, cad={cad_grid2}")

                    features2 = _fuse(features2, features2_cad)
                """
                
                # fuse CAD + GEN features
                if use_cad or use_gen:
                    F_img = features2

                    if use_cad:
                        cad_path = _cad_path_test(type_anomaly, img_test_nr)
                        cad_img = _read_rgb(cad_path)
                        cad_tensor2, cad_grid2 = model.prepare_image(cad_img)
                        F_cad = model.extract_features(cad_tensor2)

                        if tuple(cad_grid2) != tuple(grid_size2) or F_cad.shape[0] != F_img.shape[0]:
                            raise RuntimeError(f"CAD/real grid mismatch for {img_test_nr}: real={grid_size2}, cad={cad_grid2}")
                    else:
                        F_cad = np.zeros_like(F_img)

                    if use_gen:
                        gen_path = _gen_path_test(type_anomaly, img_test_nr)
                        gen_img = _read_rgb(gen_path)
                        gen_tensor2, gen_grid2 = model.prepare_image(gen_img)
                        F_gen = model.extract_features(gen_tensor2)

                        if tuple(gen_grid2) != tuple(grid_size2) or F_gen.shape[0] != F_img.shape[0]:
                            raise RuntimeError(f"GEN/real grid mismatch for {img_test_nr}: real={grid_size2}, gen={gen_grid2}")
                    else:
                        F_gen = np.zeros_like(F_img)

                    features2 = _fuse(F_img, F_cad, F_gen)
                
                
                if save_examples and idx < 3:
                    vis_image_test_background = model.get_embedding_visualization(features2, grid_size2, mask2)

                # Discard irrelevant features
                features2 = features2[mask2]

                # Compute distances to nearest neighbors in M
                if knn_metric == "L2":
                    distances, match2to1 = knn_index.search(features2, k = knn_neighbors)
                    if knn_neighbors > 1:
                        distances = distances.mean(axis=1)
                    distances = np.sqrt(distances)

                elif knn_metric == "L2_normalized":
                    faiss.normalize_L2(features2) 
                    distances, match2to1 = knn_index.search(features2, k = knn_neighbors)
                    if knn_neighbors > 1:
                        distances = distances.mean(axis=1)
                    distances = distances / 2   # equivalent to cosine distance (1 - cosine similarity)

                output_distances = np.zeros_like(mask2, dtype=float)
                output_distances[mask2] = distances.squeeze()
                d_masked = output_distances.reshape(grid_size2)
                
                # save inference time
                torch.cuda.synchronize() # Synchronize CUDA kernels before measuring time
                inf_time = time.time() - start_time
                inference_times[f"{type_anomaly}/{img_test_nr}"] = inf_time
                anomaly_scores[f"{type_anomaly}/{img_test_nr}"] = mean_top1p(output_distances.flatten())

                # Save the anomaly maps (raw as .npy or full resolution .tiff files)
                img_test_nr = img_test_nr.split(".")[0]
                if save_tiffs:
                    anomaly_map = dists2map(d_masked, image_test.shape)
                    tiff.imwrite(f"{plots_dir}/anomaly_maps/seed={seed}/{object_name}/test/{type_anomaly}/{img_test_nr}.tiff", anomaly_map)
                if save_patch_dists:
                    np.save(f"{plots_dir}/anomaly_maps/seed={seed}/{object_name}/test/{type_anomaly}/{img_test_nr}.npy", d_masked)

                # Save some example plots (3 per anomaly type)
                if save_examples and idx < 3:

                    fig, (ax1, ax2, ax3, ax4,) = plt.subplots(1, 4, figsize=(16, 4))

                    # plot test image, PCA + mask
                    ax1.imshow(image_test)
                    ax2.imshow(vis_image_test_background)  

                    # plot patch distances 
                    d_masked[~mask2.reshape(grid_size2)] = 0.0
                    plt.colorbar(ax3.imshow(d_masked), ax=ax3, fraction=0.12, pad=0.05, orientation="horizontal")
                    
                    # compute image level anomaly score (mean(top 1%) of patches = empirical tail value at risk for quantile 0.99)
                    score_top1p = mean_top1p(distances)
                    ax4.axvline(score_top1p, color='r', linestyle='dashed', linewidth=1, label=f"Anomaly Score: {score_top1p:.3f}")
                    ax4.legend()
                    ax4.hist(distances.flatten())

                    ax1.axis('off')
                    ax2.axis('off')
                    ax3.axis('off')

                    ax1.title.set_text("Test Image")
                    ax2.title.set_text("Test Image (PCA + Mask)")
                    ax3.title.set_text("Patch Distances (1NN)")
                    ax4.title.set_text("Histogram of Distances")

                    plt.suptitle(f"Object: {object_name}, Type: {type_anomaly}, img_path = ...{image_test_path[-40:]}, filtered patches (by masking)/all patches = {mask2.sum()}/{mask2.size}")

                    plt.tight_layout()
                    plt.savefig(f"{plots_dir}/{object_name}/examples/example_{type_anomaly}_{idx}.png")
                    plt.close()

    return anomaly_scores, time_memorybank, inference_times
