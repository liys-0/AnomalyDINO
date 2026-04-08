# Why Original CAD Images Degrade Anomaly Detection Performance

Based on our experimental findings on the MVTec dataset and the underlying architecture of the project (using Vision Transformers like DINOv2/DINOv3 coupled with a k-NN feature matching pipeline), here is the technical explanation for why integrating original CAD images (via `diff` or `concate` methods) degrades anomaly detection performance rather than helping it.

## 1. The "Domain Gap" in the Embedding Space
DINOv2 and DINOv3 are foundation models trained entirely on massive datasets of **natural, real-world images** (like the LVD-142M dataset). They learn to represent rich semantics: textures, lighting, shadows, material properties, and sensor noise.

* **The Problem:** CAD images are synthetic, perfectly rendered, flat, and textureless. They lack natural lighting, reflections, and camera artifacts. 
* **The Result:** When you pass a CAD image through a Vision Transformer, the extracted feature vector (`F_cad`) is pushed to a completely different, disjoint area of the high-dimensional embedding space compared to the real image (`F_img`). The model doesn't inherently understand that "this flat yellow polygon is the same object as this glossy, well-lit real yellow component."

## 2. Feature Differencing (`diff`) Captures the Wrong "Delta"
The mathematical intuition behind differencing (`F_img - F_cad`) is that subtracting the "perfect template" from the "real image" should leave behind only the anomaly or defect.

However, because of the domain gap mentioned above, the resulting feature vector is **overwhelmingly dominated by the difference between reality and a synthetic render**. The feature differences caused by real-world lighting, shadows, and background textures are massive compared to the subtle feature difference caused by a tiny scratch or missing pin. The true anomaly gets completely drowned out by the "render-to-reality" noise.

## 3. Spatial Misalignment at the Patch Level
Vision Transformers split images into small grids (e.g., 14x14 pixel patches) and compare them. For differencing or concatenating to work perfectly, the CAD image and the real camera image must be **perfectly registered** down to the exact pixel. 

* If a real object is shifted by even a few pixels, or if the camera lens introduces slight distortion that the CAD doesn't have, the spatial tokens will misalign. 
* When the k-Nearest Neighbors (kNN) algorithm compares these misaligned patches, it flags the edges of the object as highly anomalous because the real edge and the CAD edge are in different patches. This causes massive spikes in False Positives (which explains why Accuracy drops significantly when using CAD methods with higher shot counts).

## 4. The Curse of Dimensionality in Concatenation (`concate`)
When using `concat`, the features are stacked: `[F_img, F_cad]`. This doubles the dimensionality of the vectors. 

When the kNN algorithm computes the distance (L2 or Cosine) between a test image and the reference bank, it has to evaluate this doubled space. Because the `F_cad` portion of the vector contains rigid, synthetic features that don't accurately describe the real-world variations of the object, it acts as **dead weight or noise** in the distance calculation. It dilutes the highly sensitive, discriminative power that `F_img` (the natural DINO features) had on its own.

## Conclusion
In short, **foundation models like DINO are too smart for raw CAD images.** They are hyper-sensitive to textures, lighting, and real-world artifacts. By introducing a synthetic CAD image directly into the feature math, the pipeline is forced to compare a photograph to a 3D model render, which destroys the delicate thresholding needed to find microscopic real-world anomalies. 

*(Note: If CAD data must be used successfully in this pipeline in the future, it would likely require a "Domain Adaptation" step first—such as using an Image-to-Image translation network or ControlNet to render the CAD image into a photorealistic reference image before passing it to the feature extractor).*