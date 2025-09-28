# MiT_loss
**Official PyTorch implementation of "MiT Loss: Medical Image-aware Transfer-calibrated Loss for Enhanced Classification"**

This study is accepted by the Measurement Science and Technology: https://iopscience.iop.org/article/10.1088/1361-6501/ae08d8/meta

## Citation
If you think that our work is useful to your research, please cite using this BibTeX:
```bibtex
@article{10.1088/1361-6501/ae08d8,
	author={Pan, Weichao and Wang, Xu},
	title={MiT Loss: Medical Image-aware Transfer-calibrated Loss for Enhanced Classification},
	journal={Measurement Science and Technology},
	url={http://iopscience.iop.org/article/10.1088/1361-6501/ae08d8},
	year={2025},
	abstract={Medical image classification faces persistent challenges including limited annotated data, domain shift from natural images, and overconfident predictions that hinder reliability in clinical decision-making. To address these issues, we propose \textbf{M}edical \textbf{I}mage-aware \textbf{T}ransfer-calibrated Loss (\textbf{MiT Loss}), a novel transfer learning objective that integrates temperature scaling and entropy regularization to dynamically calibrate predictive confidence and promote uncertainty awareness. Unlike prior approaches that require architectural changes or auxiliary networks, MiT Loss is model-agnostic, training-only, and easily deployable. It adaptively learns temperature and entropy strength via dual-averaging and label entropy estimation, enabling plug-and-play uncertainty calibration without manual tuning.&#xD;We evaluate MiT Loss on six public datasets—\textit{BreastMNIST}, \textit{DermaMNIST}, \textit{PneumoniaMNIST}, \textit{RetinaMNIST}, \textit{PAD-UFES-20}, and \textit{CPN X-ray}—using a diverse set of CNNs, Transformers, and hybrid architectures. Results show consistent gains in accuracy, calibration, and robustness. Notably, general-purpose models fine-tuned with MiT Loss for only 10 epochs outperform specialized medical models trained for 50–150 epochs. For instance, under the same parameter quantity, BiFormer-Tiny + MiT achieved AUC values of 97.7\% and 88.7\%, respectively on DermaMNIST and RetinaMNIST, outperforming MedViTV2 and MedKAN by more than 5\%; EMO-2M, with only 2.3M parameters, achieved accuracy rates on multiple datasets that exceeded those of larger-parameter Medical Image-specific models. Grad-CAM visualizations further reveal more focused and anatomically consistent attention regions, highlighting the interpretability gains of uncertainty-aware learning.&#xD;By explicitly addressing domain mismatch and predictive overconfidence, MiT Loss offers a lightweight, effective, and generalizable solution for improving reliability and transparency in medical image classification. &#xD; Our code is publicly available at \url{https://github.com/JEFfersusu/MiT_loss}.}
}
