# Awesome-Self-Supervised-Papers

Collecting papers about Self-Supervised Learning.

**Last Update : 2020. 04. 16.**



Any contributions, comments are welcome.



## Pretraining / Feature / Representation

### Contrastive Learning

| Conference / Journal | Paper                                                        | ImageNet Acc (Top 1). |
| -------------------- | ------------------------------------------------------------ | --------------------- |
| arXiv:1807.03748     | [Representation learning with contrastive predictive coding (CPC)](https://arxiv.org/pdf/1807.03748.pdf) | -                     |
| arXiv:1911.05722     | [Momentum Contrast for Unsupervised Visual Representation Learning (MoCo)](https://arxiv.org/pdf/1911.05722.pdf) | 60.6 %                |
| arXiv:1905.09272     | [Data-Efficient Image Recognition contrastive predictive coding (CPC v2)](https://arxiv.org/pdf/1905.09272.pdf) | 63.8 %                |
| arXiv:2002.05709     | [A Simple Framework for Contrastive Learning of Visual Representations (SimCLR)](https://arxiv.org/pdf/2002.05709.pdf) | 69.3 %                |



### Image Transformation

| Conference / Journal | Paper                                                        | ImageNet Acc (Top 1). |
| -------------------- | ------------------------------------------------------------ | --------------------- |
| CVPR 2018            | [Unsupervised feature learning via non-parametric instance discrimination (NPID++)](https://arxiv.org/pdf/1805.01978.pdf) | 59.0 %                |
| ICCV 2019            | [Scaling and Benchmarking Self-Supervised Visual Representation Learning (Jigsaw)](https://arxiv.org/pdf/1905.01235.pdf) | 45.7 %                |
| arXiv:1912.01991     | [Self-Supervised Learning of Pretext-Invariant Representations (PIRL)](https://arxiv.org/pdf/1912.01991.pdf) | 63.6 %                |
| CVPR 2020            | [Steering Self-Supervised Feature Learning Beyond Local Pixel Statistics](https://arxiv.org/pdf/2004.02331.pdf) | -                     |



### Others

| Conference / Journal | Paper                                                        | Method               |
| -------------------- | ------------------------------------------------------------ | -------------------- |
| ICML 2018            | [Mutual Information Neural Estimation](https://arxiv.org/pdf/1801.04062.pdf) | Mutual Information   |
| NeurIPS 2019         | [Wasserstein Dependency Measure for Representation Learning](http://papers.nips.cc/paper/9692-wasserstein-dependency-measure-for-representation-learning.pdf) | Mutual Information   |
| ICLR 2020            | [On Mutual Information Maximization for Representation Learning](https://arxiv.org/pdf/1907.13625.pdf) | Mutual Information   |
| CVPR 2020            | [How Useful is Self-Supervised Pretraining for Visual Tasks?](https://arxiv.org/pdf/2003.14323.pdf) | -                    |
| CVPR 2020            | [Adversarial Robustness: From Self-Supervised Pre-Training to Fine-Tuning](https://arxiv.org/pdf/2003.12862.pdf) | Adversarial Training |







## Identification / Verification / Classification 

| Conference / Journal | Paper                                                        | Datasets   | Performance     |
| -------------------- | ------------------------------------------------------------ | ---------- | --------------- |
| CVPR 2020            | [Real-world Person Re-Identification via Degradation Invariance Learning](https://arxiv.org/pdf/2004.04933.pdf) | MLR-CHUK03 | Acc : 85.7(R@1) |



## Segmentation / Depth Estimation

| Conference / Journal | Paper                                                        | Datasets   | Performance     |
| -------------------- | ------------------------------------------------------------ | ---------- | --------------- |
| CVPR 2020            | [Self-supervised Equivariant Attention Mechanism for Weakly Supervised Semantic Segmentation](https://arxiv.org/pdf/2004.04581.pdf) | VOC 2012   | mIoU : 64.5     |
| CVPR 2020            | [Towards Better Generalization: Joint Depth-Pose Learning without PoseNet](https://arxiv.org/pdf/2004.01314.pdf) | KITTI 2015 | F1 : 18.05 %    |
| IROS 2020            | [Monocular Depth Estimation with Self-supervised Instance Adaptation](https://arxiv.org/pdf/2004.05821.pdf) | KITTI 2015 | Abs Rel : 0.074 |
| CVPR 2020            | [Novel View Synthesis of Dynamic Scenes with Globally Coherent Depths from a Monocular Camera](https://arxiv.org/pdf/2004.01294.pdf) | -          | -               |



## Detection / Recognition

| Conference / Journal | Paper                                                        | Datsets  | Performance   |
| -------------------- | ------------------------------------------------------------ | -------- | ------------- |
| CVPR 2020            | [Instance-aweare, Context-focused, and Memory-efficient Weakly Supervised Object Detection](https://arxiv.org/pdf/2004.04725.pdf) | VOC 2012 | AP(50) : 67.0 |



## Geneartion

| Conference / Journal | Paper                                                        | Task            |
| -------------------- | ------------------------------------------------------------ | --------------- |
| CVPR 2020            | [StyleRig: Rigging StyleGAN for 3D Control over Portrait Images](https://arxiv.org/pdf/2004.00121.pdf) | Portrait Images |



## Video

| Conference / Journal | Paper                                                        | Task                    | Datasets   | Performance    |
| -------------------- | ------------------------------------------------------------ | ----------------------- | ---------- | -------------- |
| TPAMI                | [A Review on Deep Learning Techniques for Video Prediction](https://arxiv.org/pdf/2004.05214.pdf) | Video prediction review | -          | -              |
| CVPR 2020            | [Distilled Semantics for Comprehensive Scene Understanding from Videos](https://arxiv.org/pdf/2003.14030.pdf) | Scene Understanding     | KITTI 2015 | Sq Rel : 0.748 |



## Others

| Conference / Journal | Paper                                                        | Task                | Performance              |
| -------------------- | ------------------------------------------------------------ | ------------------- | ------------------------ |
| CVPR 2020            | [Flow2Stereo: Effective Self-Supervised Learning of Optical Flow and Stereo Matching](https://arxiv.org/pdf/2004.02138.pdf) | Optical Flow        | F1 : 7.63% (KITTI 2012)  |
| CVPR 2020            | [Self-Supervised Viewpoint Learning From Image Collections](https://arxiv.org/pdf/2004.01793.pdf) | Viewpoint learning  | MAE : 4.0 (BIWI)         |
| CVPR 2020            | [Self-Supervised Scene De-occlusion](https://arxiv.org/pdf/2004.02788.pdf) | Remove occlusion    | mAP : 29.3 % (KINS)      |
| CVPR 2020            | [Distilled Semantics for Comprehensive Scene Understanding from Videos](https://arxiv.org/pdf/2003.14030.pdf) | Scene Understanding | -                        |
| CVPR 2020            | [Learning by Analogy : Reliable Supervision from Transformations for Unsupervised Optical Flow Estimation](https://arxiv.org/pdf/2003.13045.pdf) | Optical Flow        | F1 : 11.79% (KITTI 2015) |

