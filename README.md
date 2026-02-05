# Data-Centric Foundation Models in Computational Healthcare

:fire::fire::fire: **A survey on data-centric foundation models in computational healthcare**

**[Project Page](https://data-centric-fm-healthcare.github.io/)** | **[Paper [arXiv]](https://arxiv.org/abs/2401.02458)**

Last updated: 2026/02/05

:pencil: **If you find this repo helps, please kindly cite our survey, thanks!**

```bibtex
@article{zhang2024data,
  title={Data-Centric Foundation Models in Computational Healthcare: A Survey},
  author={Zhang, Yunkun and Gao, Jin and Tan, Zheling and Zhou, Lingfeng and Ding, Kexin and Zhou, Mu and Zhang, Shaoting and Wang, Dequan},
  journal={arXiv},
  year={2024},
  eprint={2401.02458},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  doi={10.48550/arXiv.2401.02458},
  url={https://arxiv.org/abs/2401.02458}
}
```

---

In this repository, we provide an up-to-date list of healthcare-related foundation models and datasets, which are also mentioned in our survey paper.

:book: **Contents**

- [Healthcare and medical foundation models](#Healthcare-and-medical-foundation-models)
  - [Language models](#Language-Models)
  - [Vision models](#Vision-Models)
  - [Vision-language models](#Vision-Language-Models)
  - [Protein and molecule models](#Protein-and-Molecule-Models)
  - [Other models](#Other-Models)
- [Datasets for foundation model](#Datasets-for-foundation-model)
  - [Text](#Text)
  - [Imaging](#Imaging)
  - [Genomics](#Genomics)
  - [Drug](#Drug)
  - [Multi-modal](#Multi-Modal)

---

## Healthcare and Medical Foundation Models

A star (*) after the pre-training data shows that the authors constructed the data with more than three sources.

### Language Models

| Model               | Subfield    | Paper                                                        | Code                                                         | Base       | Pre-Training Data    |
| :------------------ | :---------- | :----------------------------------------------------------- | :----------------------------------------------------------- | :--------- | :------------------- |
| Baichuan-M2          | Medicine    | [Baichuan-M2: Scaling Medical Capability with Large Verifier System](https://arxiv.org/abs/2509.02208) | [Github](https://github.com/baichuan-inc/Baichuan-M2-32B)    | Qwen2.5    | *                    |
| Baichuan-M1          | Medicine    | [Baichuan-M1: Pushing the Medical Capability of Large Language Models](https://arxiv.org/abs/2502.12671) | -                                                            | Transformer | 20T tokens*          |
| EHRMamba            | Clinic      | [EHRMamba: Towards Generalizable and Scalable Foundation Models for Electronic Health Records](https://proceedings.mlr.press/v259/fallahpour25a.html) | [Github](https://github.com/VectorInstitute/odyssey)         | Mamba      | MIMIC-IV             |
| MMedLM 2            | Medicine    | [Towards building multilingual language model for medicine](https://doi.org/10.1038/s41467-024-52417-z) | [Github](https://github.com/MAGIC-AI4Med/MMedLM)             | InternLM 2 | MMedC*               |
| BiMediX             | Medicine    | [BiMediX: Bilingual Medical Mixture of Experts LLM](https://doi.org/10.18653/v1/2024.findings-emnlp.989) | [Github](https://github.com/mbzuai-oryx/BiMediX)             | Mixtral    | BiMed1.3M*           |
| Me LLaMA            | Medicine    | [Me LLaMA: Foundation Large Language Models for Medical Applications](https://arxiv.org/abs/2402.12749) | [Github](https://github.com/BIDS-Xu-Lab/Me-LLaMA)            | LLaMA 2    | *                    |
| BioMistral          | Biomedicine | [BioMistral: A Collection of Open-Source Pretrained Large Language Models for Medical Domains](https://doi.org/10.18653/v1/2024.findings-acl.348) | -                                                            | Mistral    | PubMed Central       |
| PULSE               | Medicine    | -                                                            | [Github](https://github.com/openmedlab/PULSE)                | InternLM   | *                    |
| Meditron            | Medicine    | [MEDITRON-70B: Scaling Medical Pretraining for Large Language Models](https://arxiv.org/abs/2311.16079) | [Github](https://github.com/epfLLM/meditron)                 | LLaMA 2    | GAP-Replay*          |
| Taiyi               | Biomedicine | [Taiyi: A Bilingual Fine-Tuned Large Language Model for Diverse Biomedical Tasks](https://doi.org/10.1093/jamia/ocae037) | [Github](https://github.com/DUTIR-BioNLP/Taiyi-LLM)          | Qwen-7B / GLM4-9B | BigBio + CBLUE       |
| BioMedGPT           | Biomedicine | [BioMedGPT: An Open Multimodal Large Language Model for BioMedicine](https://doi.org/10.1109/jbhi.2024.3505955) | [Github](https://github.com/PharMolix/OpenBioMed)            | LLaMA 2    | S2ORC                |
| Clinical LLaMA-LoRA | Clinic      | [Parameter-Efficient Fine-Tuning of LLaMA for the Clinical Domain](https://doi.org/10.18653/v1/2024.clinicalnlp-1.9) | -                                                            | LLaMA      | MIMIC-IV             |
| Med-PaLM 2          | Clinic      | [Toward expert-level medical question answering with large language models](https://doi.org/10.1038/s41591-024-03423-7) | [Google](https://sites.research.google/med-palm/)            | PaLM 2     | MedQA                |
| PMC-LLaMA           | Medicine    | [PMC-LLaMA: toward building open-source language models for medicine](https://doi.org/10.1093/jamia/ocae045) | [Github](https://github.com/chaoyi-wu/PMC-LLaMA)             | LLaMA      | MedC                 |
| MedAlpaca           | Medicine    | [MedAlpaca -- An Open-Source Collection of Medical Conversational AI Models and Training Data](https://arxiv.org/abs/2304.08247) | [Github](https://github.com/kbressem/medAlpaca)              | LLaMA      | Medical Meadow       |
| BenTsao (HuaTuo)    | Biomedicine | [HuaTuo: Tuning LLaMA Model with Chinese Medical Knowledge](https://arxiv.org/abs/2304.06975) | [Github](https://github.com/SCIR-HI/Huatuo-Llama-Med-Chinese) | LLaMA      | CMeKG                |
| ChatDoctor          | Medicine    | [ChatDoctor: A Medical Chat Model Fine-Tuned on a Large Language Model Meta-AI (LLaMA) Using Medical Domain Knowledge](https://doi.org/10.7759/cureus.40895) | [Github](https://github.com/Kent0n-Li/ChatDoctor)            | LLaMA      | HealthCareMagic*     |
| Clinical-T5         | Clinic      | [Clinical-T5: Large Language Models Built Using Mimic Clinical Text](https://www.physionet.org/content/clinical-t5/1.0.0/) | [PhysioNet](https://www.physionet.org/content/clinical-t5/1.0.0/) | T5         | MIMIC-III + MIMIC-IV |
| Med-PaLM            | Clinic      | [Large Language Models Encode Clinical Knowledge](https://doi.org/10.1038/s41586-023-06291-2) | [Google](https://sites.research.google/med-palm)             | PaLM       | MedQA                |
| BioGPT              | Biomedicine | [BioGPT: Generative Pre-Trained Transformer for Biomedical Text Generation and Mining](https://doi.org/10.1093/bib/bbac409) | [Github](https://github.com/microsoft/BioGPT)                | GPT-2      | PubMed               |
| BioLinkBERT         | Biomedicine | [LinkBERT: Pretraining Language Models with Document Links](https://doi.org/10.18653/v1/2022.acl-long.551) | [Github](https://github.com/michiyasunaga/LinkBERT)          | BERT       | PubMed (citation links) |
| PubMedBERT          | Biomedicine | [Domain-Specific Language Model Pretraining for Biomedical Natural Language Processing](https://doi.org/10.1145/3458754) | [Microsoft](https://microsoft.github.io/BLURB/models.html)   | BERT       | PubMed               |
| BioBERT             | Biomedicine | [BioBERT: A Pre-Trained Biomedical Language Representation Model for Biomedical Text Mining](https://doi.org/10.1093/bioinformatics/btz682) | [Github](https://github.com/naver/biobert-pretrained)        | BERT       | PubMed + PMC         |
| BlueBERT            | Biomedicine | [An Empirical Study of Multi-Task Learning on BERT for Biomedical Text Mining](https://doi.org/10.18653/v1/2020.bionlp-1.22) | [Github](https://github.com/ncbi-nlp/BLUE_Benchmark)         | BERT       | PubMed + MIMIC-III   |
| Clinical BERT       | Clinic      | [Publicly Available Clinical BERT Embeddings](https://doi.org/10.18653/v1/w19-1909) | [Github](https://github.com/EmilyAlsentzer/clinicalBERT)     | BERT       | MIMIC-III            |
| SciBERT             | Biomedicine | [SciBERT: A Pretrained Language Model for Scientific Text](https://doi.org/10.18653/v1/d19-1371) | [Github](https://github.com/allenai/scibert)                 | BERT       | Semantic Scholar     |

### Vision Models

| Model           | Subfield    | Paper                                                        | Code                                                         | Base       | Pre-Training Data          |
| :-------------- | :---------- | :----------------------------------------------------------- | :----------------------------------------------------------- | :--------- | :------------------------- |
| FastGlioma       | Pathology   | [Foundation models for fast, label-free detection of glioma infiltration](https://doi.org/10.1038/s41586-024-08169-3) | -                                                            | -          | Label-free optical microscopy (4M images)* |
| MedLSAM         | Radiology   | [MedLSAM: Localize and Segment Anything Model for 3D CT Images](https://doi.org/10.1016/j.media.2024.103370) | [Github](https://github.com/openmedlab/MedLSAM)              | SAM        | *                          |
| BiomedParse     | Biomedicine | [A Foundation Model for Joint Segmentation, Detection and Recognition of Biomedical Objects across Nine Modalities](https://www.nature.com/articles/s41592-024-02499-w) | [Github](https://github.com/microsoft/BiomedParse)           | SEEM       | BiomedParseData*           |
| Universal Model | Radiology   | [Universal and Extensible Language-Vision Models for Organ Segmentation and Tumor Detection from Abdominal Computed Tomography](https://doi.org/10.1016/j.media.2024.103226) | [Github](https://github.com/ljwztc/CLIP-Driven-Universal-Model) | -          | *                          |
| CHIEF           | Pathology   | [A pathology foundation model for cancer diagnosis and prognosis prediction](https://www.nature.com/articles/s41586-024-07894-z) | [Github](https://github.com/hms-dbmi/CHIEF)                  | CTransPath | *                          |
| USFM            | Sonography  | [USFM: A Universal Ultrasound Foundation Model Generalized to Tasks and Organs towards Label Efficient Image Analysis](https://doi.org/10.1016/j.media.2024.103202) | -                                                            | MIM        | 3M-US*                     |
| BrainSegFounder | Radiology   | [BrainSegFounder: towards 3D foundation models for neuroimage segmentation](https://doi.org/10.1016/j.media.2024.103301) | [Github](https://github.com/lab-smile/BrainSegFounder)       | SwinUNETR  | UK Biobank + BraTS + ATLAS |
| MedSAM          | Medicine    | [Segment Anything in Medical Images](https://www.nature.com/articles/s41467-024-44824-z) | [Github](https://github.com/bowang-lab/MedSAM)               | SAM        | *                          |
| Prov-GigaPath   | Pathology   | [A Whole-Slide Foundation Model for Digital Pathology from Real-World Data](https://www.nature.com/articles/s41586-024-07441-w) | [Github](https://github.com/prov-gigapath/prov-gigapath)     | -          | Prov-Path*                 |
| BEPH            | Pathology   | [A Foundation Model for Generalizable Cancer Diagnosis and Survival Prediction from Histopathological Images](https://doi.org/10.1038/s41467-025-57587-y) | [Github](https://github.com/Zhcyoung/BEPH)                   | BEiTv2     | *                          |
| Pai et al.      | Radiology   | [Foundation Model for Cancer Imaging Biomarkers](https://www.nature.com/articles/s42256-024-00807-9#Sec11) | [Github](https://github.com/AIM-Harvard/foundation-cancer-image-biomarker) | SimCLR     | *                          |
| VIS-MAE         | Radiology   | [VIS-MAE: An Efficient Self-supervised Learning Approach on Medical Image Segmentation and Classification](https://doi.org/10.1007/978-3-031-73290-4_10) | -                                                            | MAE        | *                          |
| SegmentAnyBone  | Radiology   | [SegmentAnyBone: A Universal Model that Segments Any Bone at Any Location on MRI](https://doi.org/10.1016/j.media.2025.103469) | [Github](https://github.com/mazurowski-lab/SegmentAnyBone)   | SAM        | *                          |
| RudolfV         | Pathology   | [RudolfV: A Foundation Model by Pathologists for Pathologists](https://arxiv.org/abs/2401.04079) | -                                                            | DINOv2     | *                          |
| PathoDuet       | Pathology   | [PathoDuet: Foundation Models for Pathological Slide Analysis of H&E and IHC Stains](https://doi.org/10.1016/j.media.2024.103289) | [Github](https://github.com/openmedlab/PathoDuet)            | MoCo v3    | TCGA + HyReCo + BCI        |
| UNI             | Pathology   | [Towards a general-purpose foundation model for computational pathology](https://doi.org/10.1038/s41591-024-02857-3) | -                                                            | DINOv2     | Mass-100K                  |
| REMEDIS         | Radiology   | [Robust and Data-Efficient Generalization of Self-Supervised Machine Learning for Diagnostic Imaging](https://doi.org/10.1038/s41551-023-01049-7) | [Github](https://github.com/google-research/medical-ai-research-foundations) | SimCLR     | MIMIC-IV + CheXpert        |
| Virchow         | Pathology   | [A foundation model for clinical-grade computational pathology and rare cancers detection](https://doi.org/10.1038/s41591-024-03141-0) | -                                                            | DINOv2     | *                          |
| RETFound        | Retinopathy | [A Foundation Model for Generalizable Disease Detection from Retinal Images](https://www.nature.com/articles/s41586-023-06555-x) | [Github](https://github.com/rmaphoh/RETFound_MAE)            | MAE        | *                          |
| CTransPath      | Pathology   | [Transformer-Based Unsupervised Contrastive Learning for Histopathological Image Classification](https://doi.org/10.1016/j.media.2022.102559) | [Github](https://github.com/Xiyue-Wang/TransPath)            | -          | TCGA + PAIP                |
| HIPT            | Pathology   | [Scaling Vision Transformers to Gigapixel Images via Hierarchical Self-Supervised Learning](https://openaccess.thecvf.com/content/CVPR2022/html/Chen_Scaling_Vision_Transformers_to_Gigapixel_Images_via_Hierarchical_Self-Supervised_Learning_CVPR_2022_paper.html) | [Github](https://github.com/mahmoodlab/HIPT)                 | DINO       | TCGA                       |

### Vision-Language Models

| Model        | Subfield    | Paper                                                        | Code                                                         | Base             | Pre-Training Data                   |
| :----------- | :---------- | :----------------------------------------------------------- | :----------------------------------------------------------- | :--------------- | :---------------------------------- |
| TITAN        | Pathology   | [A multimodal whole-slide foundation model for pathology](https://doi.org/10.1038/s41591-025-03982-3) | -                                                            | CONCH            | 335k WSIs + reports + synthetic captions* |
| DentVLM      | Dentistry   | [DentVLM: A Multimodal Vision-Language Model for Comprehensive Dental Diagnosis and Enhanced Clinical Practice](https://arxiv.org/abs/2509.23344) | -                                                            | Qwen2-VL          | 2.4M oral images + 88.6K dental VQA*      |
| Lingshu      | Medicine    | [Lingshu: A Generalist Foundation Model for Unified Multimodal Medical Understanding and Reasoning](https://arxiv.org/abs/2506.07044) | [Project](https://alibaba-damo-academy.github.io/lingshu/)   | Qwen2.5-VL         | *                                   |
| MUSK         | Pathology   | [A Vision–Language Foundation Model for Precision Oncology](https://www.nature.com/articles/s41586-024-08378-w) | [Github](https://github.com/lilab-stanford/MUSK)             | BEiT             | *                                   |
| MONET        | Dermatology | [Transparent Medical Image AI via an Image–Text Foundation Model Grounded in Medical Literature](https://www.nature.com/articles/s41591-024-02887-x) | [Github](https://github.com/suinleelab/MONET)                | CLIP             | PubMed + textbooks                  |
| Uni-Med      | Medicine    | [Uni-Med: A Unified Medical Generalist Foundation Model For Multi-Task Learning Via Connector-MoE](https://doi.org/10.52202/079017-2582) | -                                                            | CLIP + LLaMA 2   | *                                   |
| MaCo         | Radiology   | [Enhancing Representation in Radiography-Reports Foundation Model: A Granular Alignment Algorithm Using Masked Contrastive Learning](https://doi.org/10.1038/s41467-024-51749-0) | [Github](https://github.com/SZUHvern/MaCo)                   | CLIP + MAE       | MIMIC-CXR                           |
| RadFound     | Radiology   | [Expert-Level Vision-Language Foundation Model for Real-World Radiology and Comprehensive Evaluation](https://arxiv.org/abs/2409.16183) | -                                                            | -                | RadVLCorpus*                        |
| BiomedGPT    | Biomedicine | [A Generalist Vision–Language Foundation Model for Diverse Biomedical Tasks](https://www.nature.com/articles/s41591-024-03185-2) | [Github](https://github.com/taokz/BiomedGPT)                 | -                | *                                   |
| PRISM        | Pathology   | [PRISM: A Multi-Modal Generative Foundation Model for Slide-Level Histopathology](https://arxiv.org/abs/2405.10254) | -                                                            | CoCa             | *                                   |
| Med-Gemini   | Medicine    | [Capabilities of Gemini Models in Medicine](https://arxiv.org/abs/2404.18416) | -                                                            | Gemini           | *                                   |
| EchoCLIP     | Sonography  | [Vision-Language Foundation Model for Echocardiogram Interpretation](https://www.nature.com/articles/s41591-024-02959-y) | [Github](https://github.com/echonet/echo_CLIP)               | CLIP             | *                                   |
| Med-PaLM M   | Biomedicine | [Towards Generalist Biomedical AI](https://doi.org/10.1056/AIoa2300138) | -                                                            | PaLM             | MultiMedBench*                      |
| ChemDFM      | Chemistry   | [Developing ChemDFM as a large language foundation model for chemistry](https://doi.org/10.1016/j.xcrp.2025.102523) | -                                                            | LLaMA            | PubMed + USPTO                      |
| CheXagent    | Radiology   | [A Vision-Language Foundation Model to Enhance Efficiency of Chest X-ray Interpretation](https://arxiv.org/abs/2401.12208) | [Github](https://github.com/Stanford-AIMI/CheXagent)         | BLIP-2           | CheXinstruct*                       |
| SAT          | Radiology   | [Large-vocabulary segmentation for medical images with text prompts](https://doi.org/10.1038/s41746-025-01964-w) | [Github](https://github.com/zhaoziheng/SAT)                  | -                | SAT-DS*                             |
| PathChat     | Pathology   | [Vision–language AI assistance in human pathology](https://doi.org/10.1038/s41587-024-02326-9) | -                                                            | LLaVA            | PathChatInstruct*                   |
| Qilin-Med-VL | Radiology   | [Qilin-Med-VL: Towards Chinese Large Vision-Language Model for General Healthcare](https://arxiv.org/abs/2310.17956) | [Github](https://github.com/williamliujl/Qilin-Med-VL)       | LLaVA            | Chi-Med-VL*                         |
| CXR-CLIP     | Radiology   | [CXR-CLIP: Toward Large Scale Chest X-ray Language-Image Pre-training](https://doi.org/10.1007/978-3-031-43895-0_10) | [Github](https://github.com/kakaobrain/cxr-clip)             | CLIP             | MIMIC-CXR + CheXpert + ChestX-ray14 |
| PathLDM      | Pathology   | [PathLDM: Text conditioned Latent Diffusion Model for Histopathology](https://openaccess.thecvf.com/content/WACV2024/html/Yellapragada_PathLDM_Text_Conditioned_Latent_Diffusion_Model_for_Histopathology_WACV_2024_paper.html) | [Github](https://github.com/cvlab-stonybrook/PathLDM)        | Latent Diffusion | TCGA-BRCA + GPT-3.5                 |
| RadFM        | Radiology   | [Towards generalist foundation model for radiology by leveraging web-scale 2D&3D medical data](https://doi.org/10.1038/s41467-025-62385-7) | [Github](https://github.com/chaoyi-wu/RadFM)                 | -                | MedMD*                              |
| KAD          | Radiology   | [Knowledge-Enhanced Visual-Language Pre-Training on Chest Radiology Images](https://www.nature.com/articles/s41467-023-40260-7) | [Github](https://github.com/xiaoman-zhang/KAD)               | CLIP             | MIMIC-CXR + UMLS                    |
| Med-Flamingo | Medicine    | [Med-Flamingo: A Multimodal Medical Few-Shot Learner](https://proceedings.mlr.press/v225/moor23a.html) | [Github](https://github.com/snap-stanford/med-flamingo)      | Flamingo         | MTB + PMC-OA                        |
| CONCH        | Pathology   | [A Visual-Language Foundation Model for Computational Pathology](https://www.nature.com/articles/s41591-024-02856-4) | [Github](https://github.com/mahmoodlab/CONCH)                | CoCa             | PubMed + PMC                        |
| QuiltNet     | Pathology   | [Quilt-1M: One Million Image-Text Pairs for Histopathology](https://proceedings.neurips.cc/paper_files/paper/2023/hash/775ec578876fa6812c062644964b9870-Abstract-Datasets_and_Benchmarks.html) | [Github](https://github.com/wisdomikezogwo/quilt1m)          | CLIP             | Quilt-1M*                           |
| PathAsst     | Pathology   | [PathAsst: A Generative Foundation AI Assistant towards Artificial General Intelligence of Pathology](https://doi.org/10.1609/AAAI.V38I5.28308) | [Github](https://github.com/superjamessyx/Generative-Foundation-AI-Assistant-for-Pathology) | CLIP             | PathCap + PathInstruct*             |
| PLIP         | Pathology   | [A Visual-Language Foundation Model for Pathology Image Analysis Using Medical Twitter](https://doi.org/10.1038/s41591-023-02504-3) | [Huggingface](https://huggingface.co/spaces/vinid/webplip)   | CLIP             | OpenPath*                           |
| MI-Zero      | Pathology   | [Visual Language Pretrained Multiple Instance Zero-Shot Transfer for Histopathology Images](https://openaccess.thecvf.com/content/CVPR2023/html/Lu_Visual_Language_Pretrained_Multiple_Instance_Zero-Shot_Transfer_for_Histopathology_Images_CVPR_2023_paper.html) | [Github](https://github.com/mahmoodlab/MI-Zero)              | CLIP             | ARCH                                |
| LLaVA-Med    | Biomedicine | [LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine in One Day](https://proceedings.neurips.cc/paper_files/paper/2023/hash/5abcdf8ecdcacba028c6662789194572-Abstract-Datasets_and_Benchmarks.html) | [Github](https://github.com/microsoft/LLaVA-Med)             | LLaVA            | PMC-15M + GPT-4                     |
| MedVInT      | Biomedicine | [PMC-VQA: Visual Instruction Tuning for Medical Visual Question Answering](https://arxiv.org/abs/2305.10415) | [Github](https://github.com/xiaoman-zhang/PMC-VQA)           | -                | PMC-VQA*                            |
| PMC-CLIP     | Biomedicine | [PMC-CLIP: Contrastive Language-Image Pre-Training Using Biomedical Documents](https://doi.org/10.1007/978-3-031-43993-3_51) | [Github](https://github.com/WeixiongLin/PMC-CLIP)            | CLIP             | PMC-OA*                             |
| BiomedCLIP   | Biomedicine | [A Multimodal Biomedical Foundation Model Trained from Fifteen Million Image–Text Pairs](https://doi.org/10.1056/AIoa2400640) | [Huggingface](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) | CLIP             | PMC-15M*                            |
| MedKLIP      | Radiology   | [MedKLIP: Medical Knowledge Enhanced Language-Image Pre-Training for X-ray Diagnosis](https://doi.org/10.1109/iccv51070.2023.01954) | [Github](https://github.com/MediaBrain-SJTU/MedKLIP)         | CLIP             | MIMIC-CXR                           |
| MedCLIP      | Medicine    | [MedCLIP: Contrastive Learning from Unpaired Medical Images and Text](https://doi.org/10.18653/v1/2022.emnlp-main.256) | [Github](https://github.com/RyanWangZf/MedCLIP)              | CLIP             | CheXpert + MIMIC-CXR                |
| CheXzero     | Radiology   | [Expert-Level Detection of Pathologies from Unannotated Chest X-ray Images via Self-Supervised Learning](https://www.nature.com/articles/s41551-022-00936-9) | [Github](https://github.com/rajpurkarlab/CheXzero)           | CLIP             | MIMIC-CXR                           |
| PubMedCLIP   | Radiology   | [PubMedCLIP: How Much Does CLIP Benefit Visual Question Answering in the Medical Domain?](https://doi.org/10.18653/v1/2023.findings-eacl.88) | [Github](https://github.com/sarahESL/PubMedCLIP)             | CLIP             | ROCO                                |

### Protein and Molecule Models

| Model         | Subfield   | Paper                                                        | Code                                                         | Base        | Pre-Training Data |
| :------------ | :--------- | :----------------------------------------------------------- | :----------------------------------------------------------- | :---------- | :---------------- |
| nach0         | Molecules  | [nach0: Multimodal Natural and Chemical Languages Foundation Model](https://pubs.rsc.org/en/content/articlehtml/2024/sc/d4sc00966e) | [Github](https://github.com/insilicomedicine/nach0)          | T5          | *                 |
| MoleculeSTM   | Drug       | [Multi-modal Molecule Structure-text Model for Text-based Retrieval and Editing](https://www.nature.com/articles/s42256-023-00759-6) | [Github](https://github.com/chao1224/MoleculeSTM)            | CLIP        | PubChem           |
| AlphaMissense | Proteomics | [Accurate Proteome-Wide Missense Variant Effect Prediction with AlphaMissense](https://doi.org/10.1126/science.adg7492) | [Github](https://github.com/deepmind/alphamissense)          | AlphaFold   | PDB + UniRef      |
| GET           | Genomics   | [A foundation model of transcription across human cell types](https://doi.org/10.1038/s41586-024-08391-z) | [Huggingface](https://huggingface.co/spaces/get-foundation/getdemo) | Transformer | *                 |
| GIT-Mol       | Molecules  | [GIT-Mol: A Multi-Modal Large Language Model for Molecular Science with Graph, Image, and Text](https://doi.org/10.1016/j.compbiomed.2024.108073) | [Github](https://github.com/AI-HPC-Research-Team/GIT-Mol)    | T5 + BLIP-2 | PubChem           |
| ESM-2         | Proteomics | [Evolutionary-Scale Prediction of Atomic-Level Protein Structure with a Language Model](https://doi.org/10.1126/science.ade2574) | [Github](https://github.com/facebookresearch/esm)            | Transformer | UniRef            |
| AlphaFold 2   | Proteomics | [Highly Accurate Protein Structure Prediction with AlphaFold](https://www.nature.com/articles/s41586-021-03819-2) | [Github](https://github.com/google-deepmind/alphafold)       | -           | PDB + Uniclust30  |

### Other Models

| Model     | Subfield            | Paper                                                                                                                                | Code | Base        | Pre-Training Data |
|:--------- |:------------------- |:------------------------------------------------------------------------------------------------------------------------------------ |:---- |:----------- |:----------------- |
| OmniNA    | Nucleotide sequence | [OmniNA: A Foundation Model for Nucleotide Sequences](https://www.biorxiv.org/content/10.1101/2024.01.14.575543.abstract)            | -    | LLaMA       | NCBI              |
| LaBraM    | EEG                 | [Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI](https://openreview.net/forum?id=QzTpTRVtrP) | -    | Transformer | *                 |
| Neuro-GPT | EEG                 | [Neuro-GPT: Towards A Foundation Model For EEG](https://doi.org/10.1109/ISBI56570.2024.10635453)                                                 | -    | -           | TUH EEG           |

## Datasets for Foundation Model

### Text

| Dataset (Paper)                                              | Description                                                  | Link                                                         |
| :----------------------------------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| MedBench ([DOI](https://doi.org/10.26599/bdma.2024.9020044))         | A Chinese medical LLM benchmark with 300,901 Chinese questions covering 43 clinical specialties, combined with an automatic evaluation system | [Official site](https://medbench.opencompass.org.cn)         |
| MMedBench ([DOI](https://doi.org/10.1038/s41467-024-52417-z))        | A multilingual medical QA benchmark, where questions are categorized into 21 topics | [Github](https://github.com/MAGIC-AI4Med/MMedLM)             |
| MMedC ([DOI](https://doi.org/10.1038/s41467-024-52417-z))            | A multilingual medical corpus containing over 25.5B tokens   | [Github](https://github.com/MAGIC-AI4Med/MMedLM)             |
| BiMed1.3M ([DOI](https://doi.org/10.18653/v1/2024.findings-emnlp.989))        | An English and Arabic bilingual dataset of 1.3M samples of medical QA and chat | [Github](https://github.com/mbzuai-oryx/BiMediX)             |
| GAP-Replay ([arXiv](https://arxiv.org/abs/2311.16079))       | 48.1B tokens from 4 medical corpora including guidelines, abstracts, papers, and replay | [Github](https://github.com/epfLLM/meditron)                 |
| Huatuo-26M ([DOI](https://doi.org/10.18653/v1/2025.findings-naacl.211))       | 26M Chinese medical QA pairs                                 | [Github](https://github.com/FreedomIntelligence/Huatuo-26M)  |
| Medical Meadow ([arXiv](https://arxiv.org/abs/2304.08247))   | 16M medical QA pairs collected from 9 sources                | [Github](https://github.com/kbressem/medAlpaca)              |
| MultiMedQA ([Nature](https://www.nature.com/articles/s41586-023-06291-2)) | 6 existing and 1 online-collected medical QA dataset         | [Nature](https://www.nature.com/articles/s41586-023-06291-2#data-availability) |
| BigBio ([NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2022/hash/a583d2197eafc4afdd41f5b8765555c5-Abstract-Datasets_and_Benchmarks.html)) | 126+ biomedical NLP datasets covering 13 task categories and 10+ languages | [Github](https://github.com/bigscience-workshop/biomedical)  |
| MedMCQA ([MLR](https://proceedings.mlr.press/v174/pal22a.html)) | 194K multiple-choice questions covering 2.4K healthcare topics | [Official site](https://medmcqa.github.io/)                  |
| MedQA-USMLE ([MDPI](https://www.mdpi.com/2076-3417/11/14/6421)) | 61,097 multiple choice questions based on USMLE in three languages | [Github](https://github.com/jind11/MedQA)                    |
| CBLUE ([DOI](https://doi.org/10.18653/v1/2022.acl-long.544))            | A Chinese biomedical language understanding evaluation benchmark with 18 datasets | [Official site](https://tianchi.aliyun.com/dataset/95414)    |
| BLURB ([DOI](https://doi.org/10.1145/3458754))            | 13 biomedical NLP datasets in 6 tasks                        | [Official site](https://microsoft.github.io/BLURB/index.html) |
| PubMedQA ([DOI](https://doi.org/10.18653/v1/d19-1259))         | 1K expert-annotated, 61.2K unlabeled, and 211.3K artificially generated biomedical QA instances | [Official site](https://pubmedqa.github.io/)                 |
| BLUE ([DOI](https://doi.org/10.18653/v1/w19-5006))             | 5 language tasks with 10 biomedical and clinical text datasets | [Github](https://github.com/ncbi-nlp/BLUE_Benchmark)         |
| webMedQA ([BMC](https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-019-0761-8)) | 63,284 real-world Chinese medical questions with over 300K answers | [Github](https://github.com/hejunqing/webMedQA)              |
| MedMentions ([DOI](https://doi.org/10.24432/C5G59C))      | 4,392 papers annotated by experts with mentions of UMLS entities | [Github](https://github.com/chanzuckerberg/MedMentions)      |
| MIMIC-III ([Nature](https://www.nature.com/articles/sdata201635)) | Critical care data for over 40,000 patients                  | [Official site](https://mimic.mit.edu/docs/iii/)             |
| ClinicalTrials.gov                                           | An online database of clinical research studies, including clinical trials and observational studies | [Official site](https://clinicaltrials.gov)                  |

### Imaging

| Dataset (Paper)                                              | Description                                                  | Link                                                         |
| :----------------------------------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| 3M-US                                                        | 2,187,915 ultrasound images of 12 common organs              | -                                                            |
| AbdomenAtlas ([arXiv](https://arxiv.org/abs/2407.16697))     | 20,460 3D CT volumes from 112 hospitals, with 673K masks of anatomical structures | [Github](https://github.com/MrGiovanni/AbdomenAtlas)         |
| BiomedParseData ([Nature](https://www.nature.com/articles/s41592-024-02499-w)) | 1.1M images, 3.4M image-mask-label triples, and 6.8M image-mask-description triples | [Github](https://github.com/microsoft/BiomedParse)           |
| Mass-100K ([DOI](https://doi.org/10.1038/s41591-024-02857-3))        | 100M tissue patches from 100,426 diagnostic H&E WSIs across 20 major tissue types | -                                                            |
| RETFound ([Nature](https://www.nature.com/articles/s41586-023-06555-x)) | Unannotated retinal images, containing 904,170 CFPs and 736,442 OCT scans | [Nature](https://www.nature.com/articles/s41586-023-06555-x#data-availability) |
| AbdomenAtlas-8K ([NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2023/hash/7331077e0449e94a91370c46b4f80f57-Abstract-Datasets_and_Benchmarks.html))  | 8,448 CT volumes with per-voxel annotated eight abdominal organs | [Github](https://github.com/MrGiovanni/AbdomenAtlas)         |
| Med-MNIST v2 ([Nature](https://www.nature.com/articles/s41597-022-01721-8)) | 12 2D and 6 3D datasets for biomedical image classification  | [Official site](https://medmnist.com/)                       |
| EchoNet-Dynamic ([DOI](https://doi.org/10.1038/s41586-020-2145-8)) | 10,030 expert-annotated echocardiogram videos                | [Official site](https://echonet.github.io/dynamic/)          |
| CheXpert ([DOI](https://doi.org/10.1609/AAAI.V33I01.3301590))         | 224,316 chest radiographs of 65,240 patients                 | [Official site](https://stanfordmlgroup.github.io/competitions/chexpert/) |
| Kather Colon Dataset ([PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6345440/)) | 100K histological images of human colorectal cancer and healthy tissue | [Zenodo](https://zenodo.org/records/1214456)                 |
| DeepLesion ([PMC](https://pubmed.ncbi.nlm.nih.gov/30035154/)) | 32K CT scans with annotations and semantic labels from radiological reports | [NIH](https://www.nih.gov/news-events/news-releases/nih-clinical-center-releases-dataset-32000-ct-images) |
| ChestXray-NIHCC ([DOI](https://doi.org/10.1109/CVPR.2017.369))  | 100K radiographs with labels from more than 30,000 patients  | [NIH](https://nihcc.app.box.com/v/ChestXray-NIHCC)           |
| ISIC                                                         | An archive containing 23K skin lesion images with labels & Imaging | [Official site](https://www.isic-archive.com/)               |

### Genomics

| Dataset (Paper)                                                              | Description                                                                                                   | Link                                                         |
|:---------------------------------------------------------------------------- |:------------------------------------------------------------------------------------------------------------- |:------------------------------------------------------------ |
| 1000 Genomes Project ([Nature](https://www.nature.com/articles/nature15393)) | A comprehensive catalog of human genetic variations                                                           | [Official site](https://www.internationalgenome.org/)        |
| ENCODE ([Nature](https://www.nature.com/articles/nature11247))               | A platform of genomics data and encyclopedia with integrative-level and ground-level annotations              | [NIH](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3439153/) |
| dbSNP ([NIH](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC29783/))           | A collection of human single nucleotide variations, microsatellites, and small-scale insertions and deletions | [NIH](https://pubmed.ncbi.nlm.nih.gov/11125122/)             |

### Drug

| Dataset (Paper)                                                         | Description                                                                                     | Link                                            |
|:----------------------------------------------------------------------- |:----------------------------------------------------------------------------------------------- |:----------------------------------------------- |
| DrugChat ([arXiv](https://arxiv.org/abs/2309.03907))                    | 143,517 question-answer pairs covering 10,834 drug compounds, collected from PubChem and ChEMBL | - |
| PubChem ([NIH](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9825602/))  | A collection of 900+ sources of chemical information data                                       | [NIH](https://pubchem.ncbi.nlm.nih.gov/)        |
| DrugBank ([NIH](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5753335/)) | A web-enabled structured database of molecular information about drugs                          | [Official site](https://www.drugbank.com/)      |
| ChEMBL ([NIH](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3245175/))   | 20M bioactivity measurements for 2.4M distinct compounds and 15K protein targets                | [Official site](https://www.ebi.ac.uk/chembl/)  |

### Multi-Modal

| Dataset (Paper)                                              | Description                                                  | Link                                                         |
| :----------------------------------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| MultiMedBench ([NEJM AI](https://doi.org/10.1056/AIoa2300138)) | A multi-modal benchmark comprising 12 data sources and 14 tasks | -                                                            |
| RadGenome-Chest CT ([arXiv](https://arxiv.org/abs/2404.16754)) | A dataset of 3D chest CT, including 197 organ-level segmentation masks, 665K multi-granularity grounded reports, and 1.3M grounded VQA pairs | -                                                            |
| OmniMedVQA ([DOI](https://doi.org/10.1109/CVPR52733.2024.02093))       | 131,813 question-answering items with 120,530 images from 12 modalities and 26 human anatomical regions, collected from 75 medical datasets | -                                                            |
| SAT-DS ([DOI](https://doi.org/10.1038/s41746-025-01964-w))           | 11,462 scans with 142,254 segmentation annotations spanning 8 human body regions from 31 medical image segmentation datasets, together with domain knowledge from e-Anatomy and UMLS | [Github](https://github.com/zhaoziheng/SAT)                  |
| PathChatInstruct ([DOI](https://doi.org/10.1038/s41587-024-02326-9)) | 250K+ diverse disease-agnostic visual-language instructions with image and text | -                                                            |
| Chi-Med-VL ([arXiv](https://arxiv.org/abs/2310.17956))       | 580,014 image-text pairs and 469,441 question-answer pairs for general healthcare in Chinese | [Github](https://github.com/williamliujl/Qilin-Med-VL)       |
| MedMD ([DOI](https://doi.org/10.1038/s41467-025-62385-7))            | 15.5M 2D scans and 180k 3D radiology scans  with textual descriptions | [Github](https://github.com/chaoyi-wu/RadFM)                 |
| OpenPath ([Nature](https://www.nature.com/articles/s41591-023-02504-3)) | 208,414 pathology images paired with natural language descriptions | [Huggingface](https://huggingface.co/spaces/vinid/webplip)   |
| Quilt-1M ([NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2023/hash/775ec578876fa6812c062644964b9870-Abstract-Datasets_and_Benchmarks.html))         | 1M image-text pairs for histopathology                       | [Github](https://github.com/wisdomikezogwo/quilt1m)          |
| Med-MMHL ([arXiv](https://arxiv.org/abs/2306.08871))         | Human- and LLM-generated misinformation detection dataset    | [Github](https://github.com/styxsys0927/Med-MMHL)            |
| Mol-Instructions ([OpenReview](https://openreview.net/forum?id=Tlsdsb6l9n)) | 148K molecule-oriented, 505K protein-oriented, and biomolecular text instructions | [Huggingface](https://huggingface.co/datasets/zjunlp/Mol-Instructions) |
| PathInstruct ([DOI](https://doi.org/10.1609/AAAI.V38I5.28308))     | 180K samples of LLM-generated instruction-following data     | [Github](https://github.com/superjamessyx/Generative-Foundation-AI-Assistant-for-Pathology) |
| PMC-VQA ([arXiv](https://arxiv.org/abs/2305.10415))          | 227K VQA pairs of 149K images of various modalities or diseases | [Github](https://github.com/xiaoman-zhang/PMC-VQA)           |
| PMC-OA ([DOI](https://doi.org/10.1007/978-3-031-43993-3_51))           | 1.6M fine-grained biomedical image-text pairs                | [Github](https://github.com/WeixiongLin/PMC-CLIP)            |
| PathCap ([DOI](https://doi.org/10.1007/978-3-031-43993-3_51))          | 142K pathology image-caption pairs from various sources      | [Github](https://github.com/WeixiongLin/PMC-CLIP)            |
| SwissProtCLAP ([DOI](https://doi.org/10.1038/s42256-025-01011-z))    | 441K text-protein sequence pairs                             | [Github](https://github.com/chao1224/chatdrug)               |
| MIMIC-IV ([Nature](https://www.nature.com/articles/s41597-022-01899-x)) | Clinical information for hospital stays of over 60,000 patients | [Official site](https://mimic.mit.edu/docs/iv/)              |
| MIMIC-CXR ([Nature](https://www.nature.com/articles/s41597-019-0322-0)) | 227,835 chest imaging studies with free-text reports for 65,379 patients | [PhysioNet](https://physionet.org/content/mimic-cxr/2.0.0/)  |
| TCGA                                                         | A landmark cancer genomics program, molecularly characterized over 20,000 primary cancer and matched normal samples spanning 33 cancer types | [Official site](https://www.cancer.gov/ccg/research/genome-sequencing/tcga) |
