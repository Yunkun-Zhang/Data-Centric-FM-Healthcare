# Data-Centric Foundation Models in Computational Healthcare

:fire::fire::fire: **A survey on data-centric foundation models in computational healthcare**

**[Project Page](https://data-centric-fm-healthcare.github.io/)** | **[Paper [arXiv]](https://arxiv.org/abs/2401.02458)**

Last updated: 2024/03/03

:pencil: **If you find this repo helps, please kindly cite our survey, thanks!**

```
@article{zhang2024data,
  title={Data-Centric Foundation Models in Computational Healthcare: A Survey},
  author={Zhang, Yunkun and Gao, Jin and Tan, Zheling and Zhou, Lingfeng and Ding, Kexin and Zhou, Mu and Zhang, Shaoting and Wang, Dequan},
  journal={arXiv preprint arXiv:2401.02458},
  year={2024}
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
| MMedLM 2            | Medicine    | [Towards Building Multilingual Language Model for Medicine](https://arxiv.org/abs/2402.13963) | [Github](https://github.com/MAGIC-AI4Med/MMedLM)             | InternLM 2 | MMedC*               |
| Me LLaMA            | Medicine    | [Me LLaMA: Foundation Large Language Models for Medical Applications](https://arxiv.org/abs/2402.12749) | [Github](https://github.com/BIDS-Xu-Lab/Me-LLaMA)            | LLaMA 2    | *                    |
| BioMistral          | Biomedicine | [BioMistral: A Collection of Open-Source Pretrained Large Language Models for Medical Domains](https://arxiv.org/abs/2402.10373) | -                                                            | Mistral    | PMC                  |
| PULSE               | Medicine    | -                                                            | [Github](https://github.com/openmedlab/PULSE)                | InternLM   | *                    |
| Meditron            | Medicine    | [Meditron-70B: Scaling Medical Pretraining for Large Language Models](https://arxiv.org/abs/2311.16079) | [Github](https://github.com/epfLLM/meditron)                 | LLaMA 2    | GAP-Replay*          |
| Taiyi               | Biomedicine | [Taiyi: A Bilingual Fine-Tuned Large Language Model for Diverse Biomedical Tasks](https://arxiv.org/abs/2311.11608) | [Github](https://github.com/DUTIR-BioNLP/Taiyi-LLM)          | Qwen       | BigBio + CBLUE       |
| BioMedGPT           | Biomedicine | [BioMedGPT: Open Multimodal Generative Pre-trained Transformer for BioMedicine](https://arxiv.org/abs/2308.09442) | [Github](https://github.com/PharMolix/OpenBioMed)            | LLaMA 2    | S2ORC                |
| Clinical LLaMA-LoRA | Clinic      | [Parameter-Efficient Fine-Tuning of LLaMA for the Clinical Domain](https://arxiv.org/abs/2307.03042) | -                                                            | LLaMA      | MIMIC-IV             |
| Med-PaLM 2          | Clinic      | [Towards Expert-Level Medical Question Answering with Large Language Models](https://arxiv.org/abs/2305.09617) | [Google](https://sites.research.google/med-palm/)            | PaLM 2     | MultiMedQA           |
| PMC-LLaMA           | Medicine    | [PMC-LLaMA: Towards Building Open-source Language Models for Medicine](https://arxiv.org/abs/2304.14454) | [Github](https://github.com/chaoyi-wu/PMC-LLaMA)             | LLaMA      | MedC                 |
| MedAlpaca           | Medicine    | [MedAlpaca -- An Open-Source Collection of Medical Conversational AI Models and Training Data](https://arxiv.org/abs/2304.08247) | [Github](https://github.com/kbressem/medAlpaca)              | LLaMA      | Medical Meadow       |
| BenTsao (HuaTuo)    | Biomedicine | [HuaTuo: Tuning LLaMA Model with Chinese Medical Knowledge](https://arxiv.org/abs/2304.06975) | [Github](https://github.com/SCIR-HI/Huatuo-Llama-Med-Chinese) | LLaMA      | CMeKG                |
| ChatDoctor          | Medicine    | [ChatDoctor: A Medical Chat Model Fine-Tuned on a Large Language Model Meta-AI (LLaMA) Using Medical Domain Knowledge](https://arxiv.org/abs/2303.14070) | [Github](https://github.com/Kent0n-Li/ChatDoctor)            | LLaMA      | HealthCareMagic*     |
| Clinical-T5         | Clinic      | [Clinical-T5: Large Language Models Built Using Mimic Clinical Text](https://www.physionet.org/content/clinical-t5/1.0.0/) | [PhysioNet](https://www.physionet.org/content/clinical-t5/1.0.0/) | T5         | MIMIC-III + MIMIC-IV |
| Med-PaLM            | Clinic      | [Large Language Models Encode Clinical Knowledge](https://arxiv.org/abs/2212.13138) | [Google](https://sites.research.google/med-palm)             | PaLM       | MultiMedQA           |
| BioGPT              | Biomedicine | [BioGPT: Generative Pre-Trained Transformer for Biomedical Text Generation and Mining](https://academic.oup.com/bib/article-abstract/23/6/bbac409/6713511) | [Github](https://github.com/microsoft/BioGPT)                | GPT-2      | PubMed               |
| BioLinkBERT         | Biomedicine | [Linkbert: Pretraining Language Models with Document Links](https://arxiv.org/abs/2203.15827) | [Github](https://github.com/michiyasunaga/LinkBERT)          | BERT       | PubMed               |
| PubMedBERT          | Biomedicine | [Domain-Specific Language Model Pretraining for Biomedical Natural Language Processing](https://arxiv.org/abs/2007.15779) | [Microsoft](https://microsoft.github.io/BLURB/models.html)   | BERT       | PubMed               |
| BioBERT             | Biomedicine | [BioBERT: A Pre-Trained Biomedical Language Representation Model for Biomedical Text Mining](https://academic.oup.com/bioinformatics/article-abstract/36/4/1234/5566506) | [Github](https://github.com/naver/biobert-pretrained)        | BERT       | PubMed + PMC         |
| BlueBERT            | Biomedicine | [An Empirical Study of Multi-Task Learning on BERT for Biomedical Text Mining](https://arxiv.org/abs/2005.02799) | [Github](https://github.com/ncbi-nlp/BLUE_Benchmark)         | BERT       | PubMed + MIMIC-III   |
| Clinical BERT       | Clinic      | [Publicly Available Clinical BERT Embeddings](https://arxiv.org/abs/1904.03323) | [Github](https://github.com/EmilyAlsentzer/clinicalBERT)     | BERT       | MIMIC-III            |
| SciBERT             | Biomedicine | [SciBERT: A Pretrained Language Model for Scientific Text](https://arxiv.org/abs/1903.10676) | [Github](https://github.com/allenai/scibert)                 | BERT       | Semantic Scholar     |

### Vision Models

| Model      | Subfield    | Paper                                                        | Code                                                         | Base   | Pre-Training Data   |
| :--------- | :---------- | :----------------------------------------------------------- | :----------------------------------------------------------- | :----- | :------------------ |
| VISION-MAE | Radiology   | [VISION-MAE: A Foundation Model for Medical Image Segmentation and Classification](https://arxiv.org/abs/2402.01034) | -                                                            | MAE    | *                   |
| RudolfV    | Pathology   | [RudolfV: A Foundation Model by Pathologists for Pathologists](https://arxiv.org/abs/2401.04079) | -                                                            | DINOv2 | *                   |
| UNI        | Pathology   | [A General-Purpose Self-Supervised Model for Computational Pathology](https://arxiv.org/abs/2308.15474) | -                                                            | DINOv2 | Mass-100K           |
| REMEDIS    | Radiology   | [Robust and Data-Efficient Generalization of Self-Supervised Machine Learning for Diagnostic Imaging](https://idp.nature.com/authorize/casa?redirect_uri=https://www.nature.com/articles/s41551-023-01049-7&casa_token=jsWqfcJssI0AAAAA:zt3n5PYal2WyePCxeKXW4q4x0gmqtWQYHCLqXbLQhK1ERML3pgp68Q7GBN1wVK9MYP5iyxBzlsaD1Tygag) | [Github](https://github.com/google-research/medical-ai-research-foundations) | SimCLR | MIMIC-IV + CheXpert |
| Virchow    | Pathology   | [Virchow: A Million-Slide Digital Pathology Foundation Model](https://arxiv.org/abs/2309.07778) | -                                                            | DINOv2 | *                   |
| RETFound   | Retinopathy | [A Foundation Model for Generalizable Disease Detection from Retinal Images](https://www.nature.com/articles/s41586-023-06555-x) | [Github](https://github.com/rmaphoh/RETFound_MAE)            | MAE    | *                   |
| CTransPath | Pathology   | [Transformer-Based Unsupervised Contrastive Learning for Histopathological Image Classification](https://www.sciencedirect.com/science/article/pii/S1361841522002043?casa_token=YBbUxnv_qsAAAAAA:YrgecQ6ecLad4Bj3JfGl0SZvjRgSQBZ27KYtpH6jU3vy6j-8hGrnQzbVFWCg0vH9Pn7r5H1Cxw) | [Github](https://github.com/Xiyue-Wang/TransPath)            | -      | TCGA + PAIP         |
| HIPT       | Pathology   | [Scaling Vision Transformers to Gigapixel Images via Hierarchical Self-Supervised Learning](https://openaccess.thecvf.com/content/CVPR2022/html/Chen_Scaling_Vision_Transformers_to_Gigapixel_Images_via_Hierarchical_Self-Supervised_Learning_CVPR_2022_paper.html?trk=public_post_comment-text) | [Github](https://github.com/mahmoodlab/HIPT)                 | DINO   | TCGA                |

### Vision-Language Models

| Model        | Subfield    | Paper                                                        | Code                                                         | Base             | Pre-Training Data                   |
| :----------- | :---------- | :----------------------------------------------------------- | :----------------------------------------------------------- | :--------------- | :---------------------------------- |
| ChemDFM      | Chemistry   | [ChemDFM: Dialogue Foundation Model for Chemistry](https://arxiv.org/abs/2401.14818) | -                                                            | LLaMA            | PubMed + USPTO                      |
| CheXagent    | Radiology   | [CheXagent: Towards a Foundation Model for Chest X-Ray Interpretation](https://arxiv.org/abs/2401.12208) | [Github](https://github.com/Stanford-AIMI/CheXagent)         | BLIP-2           | CheXinstruct*                       |
| SAT          | Radiology   | [One Model to Rule them All: Towards Universal Segmentation for Medical Images with Text Prompts](https://arxiv.org/abs/2312.17183) | [Github](https://github.com/zhaoziheng/SAT)                  | -                | SAT-DS*                             |
| PathChat     | Pathology   | [A Foundational Multimodal Vision Language AI Assistant for Human Pathology](https://arxiv.org/abs/2312.07814) | -                                                            | LLaVA            | PathChatInstruct*                   |
| Qilin-Med-VL | Radiology   | [Qilin-Med-VL: Towards Chinese Large Vision-Language Model for General Healthcare](https://arxiv.org/abs/2310.17956) | [Github](https://github.com/williamliujl/Qilin-Med-VL)       | LLaVA            | Chi-Med-VL*                         |
| CXR-CLIP     | Radiology   | [CXR-CLIP: Toward Large Scale Chest X-ray Language-Image Pre-training](https://arxiv.org/abs/2310.13292) | [Github](https://github.com/kakaobrain/cxr-clip)             | CLIP             | MIMIC-CXR + CheXpert + ChestX-ray14 |
| MaCo         | Radiology   | [Enhancing Representation in Radiography-Reports Foundation Model: A Granular Alignment Algorithm Using Masked Contrastive Learning](https://arxiv.org/abs/2309.05904) | -                                                            | MAE + CLIP       | MIMIC-CXR                           |
| PathLDM      | Pathology   | [PathLDM: Text conditioned Latent Diffusion Model for Histopathology](https://openaccess.thecvf.com/content/WACV2024/html/Yellapragada_PathLDM_Text_Conditioned_Latent_Diffusion_Model_for_Histopathology_WACV_2024_paper.html) | [Github](https://github.com/cvlab-stonybrook/PathLDM)        | Latent Diffusion | TCGA-BRCA + GPT-3.5                 |
| RadFM        | Radiology   | [Towards Generalist Foundation Model for Radiology](https://arxiv.org/abs/2308.02463) | [Github](https://github.com/chaoyi-wu/RadFM)                 | -                | MedMD*                              |
| KAD          | Radiology   | [Knowledge-Enhanced Visual-Language Pre-Training on Chest Radiology Images](https://www.nature.com/articles/s41467-023-40260-7) | [Github](https://github.com/xiaoman-zhang/KAD)               | CLIP             | MIMIC-CXR + UMLS                    |
| Med-Flamingo | Medicine    | [Med-Flamingo: A Multimodal Medical Few-Shot Learner](https://proceedings.mlr.press/v225/moor23a.html) | [Github](https://github.com/snap-stanford/med-flamingo)      | Flamingo         | MTB + PMC-OA                        |
| CONCH        | Pathology   | [Towards a Visual-Language Foundation Model for Computational Pathology](https://arxiv.org/abs/2307.12914) | -                                                            | CoCa             | PubMed + PMC                        |
| QuiltNet     | Pathology   | [Quilt-1M: One Million Image-Text Pairs for Histopathology](https://arxiv.org/abs/2306.11207) | [Github](https://github.com/wisdomikezogwo/quilt1m)          | CLIP             | Quilt-1M*                           |
| PathAsst     | Pathology   | [PathAsst: Redefining Pathology through Generative Foundation AI Assistant for Pathology](https://arxiv.org/abs/2305.15072) | [Github](https://github.com/superjamessyx/Generative-Foundation-AI-Assistant-for-Pathology) | CLIP             | PathCap + PathInstruct*             |
| PLIP         | Pathology   | [A Visual-Language Foundation Model for Pathology Image Analysis Using Medical Twitter](https://idp.nature.com/authorize/casa?redirect_uri=https://www.nature.com/articles/s41591-023-02504-3&casa_token=cnEpAWMo9RIAAAAA:_v3_yKPcr_afGn_MCirdOLLHyC63vSFVuvqu2sM4lnxJaZVQF7gmZsEjP2-W-CTQ9Xr2OVOpQEjgdIf9Jw) | [Huggingface](https://huggingface.co/spaces/vinid/webplip)   | CLIP             | OpenPath*                           |
| MI-Zero      | Pathology   | [Visual Language Pretrained Multiple Instance Zero-Shot Transfer for Histopathology Images](http://openaccess.thecvf.com/content/CVPR2023/html/Lu_Visual_Language_Pretrained_Multiple_Instance_Zero-Shot_Transfer_for_Histopathology_Images_CVPR_2023_paper.html) | [Github](https://github.com/mahmoodlab/MI-Zero)              | CLIP             | ARCH                                |
| LLaVA-Med    | Biomedicine | [LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine in One Day](https://arxiv.org/abs/2306.00890) | [Github](https://github.com/microsoft/LLaVA-Med)             | LLaVA            | PMC-15M + GPT-4                     |
| MedVInT      | Biomedicine | [PMC-VQA: Visual Instruction Tuning for Medical Visual Question Answering](https://arxiv.org/abs/2305.10415) | [Github](https://github.com/xiaoman-zhang/PMC-VQA)           | -                | PMC-VQA*                            |
| PMC-CLIP     | Biomedicine | [PMC-CLIP: Contrastive Language-Image Pre-Training Using Biomedical Documents](https://arxiv.org/abs/2303.07240) | [Github](https://github.com/WeixiongLin/PMC-CLIP)            | CLIP             | PMC-OA*                             |
| BiomedCLIP   | Biomedicine | [Large-Scale Domain-Specific Pretraining for Biomedical Vision-Language Processing](https://arxiv.org/abs/2303.00915) | [Huggingface](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) | CLIP             | PMC-15M*                            |
| MedKLIP      | Radiology   | [MedKLIP: Medical Knowledge Eenhanced Language-Image Pre-Training](https://www.medrxiv.org/content/10.1101/2023.01.10.23284412.abstract) | [Github](https://github.com/MediaBrain-SJTU/MedKLIP)         | CLIP             | MIMIC-CXR                           |
| MedCLIP      | Medicine    | [MedCLIP: Contrastive Learning from Unpaired Medical Images and Text](https://arxiv.org/abs/2210.10163) | [Github](https://github.com/RyanWangZf/MedCLIP)              | CLIP             | CheXpert + MIMIC-CXR                |
| CheXzero     | Radiology   | [Expert-Level Detection of Pathologies from Unannotated Chest X-ray Images via Self-Supervised Learning](https://www.nature.com/articles/s41551-022-00936-9) | [Github](https://github.com/rajpurkarlab/CheXzero)           | CLIP             | MIMIC-CXR                           |
| PubMedCLIP   | Radiology   | [Does CLIP Benefit Visual Question Answering in the Medical Domain as Much as it Does in the General Domain?](https://arxiv.org/abs/2112.13906) | [Github](https://github.com/sarahESL/PubMedCLIP)             | CLIP             | ROCO                                |

### Protein and Molecule Models

| Model         | Subfield   | Paper                                                        | Code                                                         | Base        | Pre-Training Data |
| :------------ | :--------- | :----------------------------------------------------------- | :----------------------------------------------------------- | :---------- | :---------------- |
| MoleculeSTM   | Drug       | [Multi-modal Molecule Structure-text Model for Text-based Retrieval and Editing](https://www.nature.com/articles/s42256-023-00759-6) | [Github](https://github.com/chao1224/MoleculeSTM)            | CLIP        | PubChem           |
| AlphaMissense | Proteomics | [Accurate Proteome-Wide Missense Variant Effect Prediction with AlphaMissense](https://www.science.org/doi/abs/10.1126/science.adg7492) | [Github](https://github.com/deepmind/alphamissense)          | AlphaFold   | PDB + UniRef      |
| GET           | Genomics   | [GET: A Foundation Model of Transcription across Human Cell Types](https://www.biorxiv.org/content/10.1101/2023.09.24.559168.abstract) | [Huggingface](https://huggingface.co/spaces/get-foundation/getdemo) | Transformer | *                 |
| GIT-Mol       | Molecules  | [GIT-Mol: A Multi-Modal Large Language Model for Molecular Science with Graph, Image, and Text](https://www.sciencedirect.com/science/article/pii/S0010482524001574?casa_token=Dkncjjih45UAAAAA:wJyM-lr4S-KMG2iqc3YkRpuHaMhJzFidXKt0PCwgJLjTQuFLN-DVA4t6CE9pTtuadXKTAe7jdeI) | [Github](https://github.com/AI-HPC-Research-Team/GIT-Mol)    | T5 + BLIP-2 | PubChem           |
| ESM-2         | Proteomics | [Evolutionary-Scale Prediction of Atomic-Level Protein Structure with a Language Model](https://www.science.org/doi/abs/10.1126/science.ade2574?casa_token=Qvgo8ZWhDYwAAAAA:SbKFf-TJQHVPNS_peeNUOoKxnsYgvp-0PMaPG1Oh5zGLrs1zdoSJBTe_qDl4n9loA7-RFE5GDJ2_kIA) | [Github](https://github.com/facebookresearch/esm)            | Transformer | UniRef            |
| AlphaFold 2   | Proteomics | [Highly Accurate Protein Structure Prediction with AlphaFold](https://www.nature.com/articles/s41586-021-03819-2) | [Github](https://github.com/google-deepmind/alphafold)       | -           | PDB + Uniclust30  |

### Other Models

| Model     | Subfield            | Paper                                                        | Code | Base        | Pre-Training Data |
| :-------- | :------------------ | :----------------------------------------------------------- | :--- | :---------- | :---------------- |
| OmniNA    | Nucleotide sequence | [OmniNA: A Foundation Model for Nucleotide Sequences](https://www.biorxiv.org/content/10.1101/2024.01.14.575543.abstract) | -    | LLaMA       | NCBI              |
| LaBraM    | EEG                 | [Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI](https://openreview.net/forum?id=QzTpTRVtrP) | -    | Transformer | *                 |
| Neuro-GPT | EEG                 | [Neuro-GPT: Developing A Foundation Model for EEG](https://arxiv.org/abs/2311.03764) | -    | -           | TUH EEG           |



## Datasets for Foundation Model

### Text

| Dataset (Paper)                                              | Description                                                  | Link                                                         |
| :----------------------------------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| MMedBench ([arXiv](https://arxiv.org/abs/2402.13963))        | A multilingual medical QA benchmark, where questions are categorized into 21 topics | [Github](https://github.com/MAGIC-AI4Med/MMedLM)             |
| MMedC ([arXiv](https://arxiv.org/abs/2402.13963))            | A multilingual medical corpus containing over 25.5B tokens   | [Github](https://github.com/MAGIC-AI4Med/MMedLM)             |
| GAP-Replay ([arXiv](https://arxiv.org/abs/2311.16079))       | 48.1B tokens from 4 medical corpora including guidelines, abstracts, papers, and replay | [Github](https://github.com/epfLLM/meditron)                 |
| Huatuo-26M ([arXiv](https://arxiv.org/abs/2305.01526))       | 26M Chinese medical QA pairs                                 | [Github](https://github.com/FreedomIntelligence/Huatuo-26M)  |
| Medical Meadow ([arXiv](https://arxiv.org/abs/2304.08247))   | 16M medical QA pairs collected from 9 sources                | [Github](https://github.com/kbressem/medAlpaca)              |
| MultiMedQA ([Nature](https://www.nature.com/articles/s41586-023-06291-2)) | 6 existing and 1 online-collected medical QA dataset         | [Nature](https://www.nature.com/articles/s41586-023-06291-2#data-availability) |
| BigBio ([Nature](https://proceedings.neurips.cc/paper_files/paper/2022/hash/a583d2197eafc4afdd41f5b8765555c5-Abstract-Datasets_and_Benchmarks.html)) | 126+ biomedical NLP datasets covering 13 task categories and 10+ languages | [Github](https://github.com/bigscience-workshop/biomedical)  |
| MedMCQA ([MLR](https://proceedings.mlr.press/v174/pal22a.html)) | 194K multiple-choice questions covering 2.4K healthcare topics | [Official site](https://medmcqa.github.io/)                  |
| MedQA-USMLE ([MDPI](https://www.mdpi.com/2076-3417/11/14/6421)) | 61,097 multiple choice questions based on USMLE in three languages | [Github](https://github.com/jind11/MedQA)                    |
| CBLUE ([arXiv](https://arxiv.org/abs/2106.08087))            | A Chinese biomedical language understanding evaluation benchmark with 18 datasets | [Official site](https://tianchi.aliyun.com/dataset/95414)    |
| BLURB ([arXiv](https://arxiv.org/abs/2007.15779))            | 13 biomedical NLP datasets in 6 tasks                        | [Official site](https://microsoft.github.io/BLURB/index.html) |
| PubMedQA ([arXiv](https://arxiv.org/abs/1909.06146))         | 1K expert-annotated, 61.2K unlabeled, and 211.3K artificially generated biomedical QA instances | [Official site](https://pubmedqa.github.io/)                 |
| BLUE ([arXiv](https://arxiv.org/abs/1906.05474))             | 5 language tasks with 10 biomedical and clinical text datasets | [Github](https://github.com/ncbi-nlp/BLUE_Benchmark)         |
| webMedQA ([BMC](https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-019-0761-8)) | 63,284 real-world Chinese medical questions with over 300K answers | [Github](https://github.com/hejunqing/webMedQA)              |
| MedMentions ([arXiv](https://arxiv.org/abs/1902.09476))      | 4,392 papers annotated by experts with mentions of UMLS entities | [Github](https://github.com/chanzuckerberg/MedMentions)      |
| MIMIC-III ([Nature](https://www.nature.com/articles/sdata201635)) | Critical care data for over 40,000 patients                  | [Official site](https://mimic.mit.edu/docs/iii/)             |
| ClinicalTrials.gov                                           | An online database of clinical research studies, including clinical trials and observational studies | [Official site](https://clinicaltrials.gov)                  |

### Imaging

| Dataset (Paper)                                              | Description                                                  | Link                                                         |
| :----------------------------------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| Mass-100K ([arXiv](https://arxiv.org/abs/2308.15474))        | 100M tissue patches from 100,426 diagnostic H&E WSIs accross 20 major tissue types | -                                                            |
| RETFound ([Nature](https://www.nature.com/articles/s41586-023-06555-x)) | Unannotated retinal images, containing 904,170 CFPs and 736,442 OCT scans | [Nature](https://www.nature.com/articles/s41586-023-06555-x#data-availability) |
| AbdomenAtlas-8K ([arXiv](https://arxiv.org/abs/2305.09666))  | 8,448 CT volumes with per-voxel annotated eight abdominal organs | [Github](https://github.com/MrGiovanni/AbdomenAtlas)         |
| Med-MNIST v2 ([Nature](https://www.nature.com/articles/s41597-022-01721-8)) | 12 2D and 6 3D datasets for biomedical image classification  | [Official site](https://medmnist.com/)                       |
| EchoNet-Dynamic ([Nature](https://idp.nature.com/authorize/casa?redirect_uri=https://www.nature.com/articles/s41586-020-2145-8&casa_token=uE_JgWrZ_UYAAAAA:9qEia-_2_fIZMgmhK0OamD4a6iq_wxaObBvA2Cp7r6criIIybpxDrwu5DLB37b2R5nZGkO1GDDa7PN6CUTQ)) | 10,030 expert-annotated echocardiogram videos                | [Official site](https://echonet.github.io/dynamic/)          |
| CheXpert ([arXiv](https://arxiv.org/abs/1901.07031))         | 224,316 chest radiographs of 65,240 patients                 | [Official site](https://stanfordmlgroup.github.io/competitions/chexpert/) |
| Kather Colon Dataset ([PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6345440/)) | 100K histological images of human colorectal cancer and healthy tissue | [Zenodo](https://zenodo.org/records/1214456)                 |
| DeepLesion ([PMC](https://pubmed.ncbi.nlm.nih.gov/30035154/)) | 32K CT scans with annotations and semantic labels from radiological reports | [NIH](https://www.nih.gov/news-events/news-releases/nih-clinical-center-releases-dataset-32000-ct-images) |
| ChestXray-NIHCC ([arXiv](https://arxiv.org/abs/1705.02315))  | 100K radiographs with labels from more than 30,000 patients  | [NIH](https://nihcc.app.box.com/v/ChestXray-NIHCC)           |
| ISIC                                                         | An archive containing 23K skin lesion images with labels & Imaging | [Official site](https://www.isic-archive.com/)               |

### Genomics

| Dataset (Paper)                                              | Description                                                  | Link                                                         |
| :----------------------------------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| 1000 Genomes Project ([Nature](https://www.nature.com/articles/nature15393)) | A comprehensive catalog of human genetic variations          | [Official site](https://www.internationalgenome.org/)        |
| ENCODE ([Nature](https://www.nature.com/articles/nature11247)) | A platform of genomics data and encyclopedia with integrative-level and ground-level annotations | [NIH](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3439153/) |
| dbSNP ([NIH](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC29783/)) | A collection of human single nucleotide variations, microsatellites, and small-scale insertions and deletions | [NIH](https://pubmed.ncbi.nlm.nih.gov/11125122/)             |

### Drug

| Dataset (Paper)                                              | Description                                                  | Link                                            |
| :----------------------------------------------------------- | :----------------------------------------------------------- | :---------------------------------------------- |
| DrugChat ([arXiv](https://arxiv.org/abs/2309.03907))         | 143,517 question-answer pairs covering 10,834 drug compounds, collected from PubChem and ChEMBL | [Github](https://github.com/UCSD-AI4H/drugchat) |
| PubChem ([NIH](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9825602/)) | A collection of 900+ sources of chemical information data    | [NIH](https://pubchem.ncbi.nlm.nih.gov/)        |
| DrugBank ([NIH](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5753335/)) | A web-enabled structured database of molecular information about drugs | [Official site](https://www.drugbank.com/)      |
| ChEMBL ([NIH](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3245175/)) | 20M bioactivity measurements for 2.4M distinct compounds and 15K protein targets | [Official site](https://www.ebi.ac.uk/chembl/)  |

### Mulit-Modal

| Dataset (Paper)                                              | Description                                                  | Link                                                         |
| :----------------------------------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| OmniMedVQA ([arXiv](https://arxiv.org/abs/2402.09181))       | 131,813 question-answering items with 120,530 images from 12 modalities and 26 human anatomical regions, collected from 75 medical datasets | -                                                            |
| SAT-DS ([arXiv](https://arxiv.org/abs/2312.17183))           | 11,462 scans with 142,254 segmentation annotations spanning 8 human body regions from 31 medical image segmentation datasets, together with domain knowledge from e-Anatomy and UMLS | [Github](https://github.com/zhaoziheng/SAT)                  |
| PathChatInstruct ([arXiv](https://arxiv.org/abs/2312.07814)) | 257,004 instructions of pathology-specific queries with image and text | -                                                            |
| Chi-Med-VL ([arXiv](https://arxiv.org/abs/2310.17956))       | 580,014 image-text pairs and 469,441 question-answer pairs for general healthcare in Chinese | [Github](https://github.com/williamliujl/Qilin-Med-VL)       |
| MedMD ([arXiv](https://arxiv.org/abs/2308.02463))            | 15.5M 2D scans and 180k 3D radiology scans  with textual descriptions | [Github](https://github.com/chaoyi-wu/RadFM)                 |
| OpenPath ([Nature](https://www.nature.com/articles/s41591-023-02504-3)) | 208,414 pathology images paired with natural language descriptions | [Huggingface](https://huggingface.co/spaces/vinid/webplip)   |
| Quilt-1M ([arXiv](https://arxiv.org/abs/2306.11207))         | 1M image-text pairs for histopathology                       | [Github](https://github.com/wisdomikezogwo/quilt1m)          |
| Med-MMHL ([arXiv](https://arxiv.org/abs/2306.08871))         | Human- and LLM-generated misinformation detection dataset    | [Github](https://github.com/styxsys0927/Med-MMHL)            |
| PathInstruct ([arXiv](https://arxiv.org/abs/2305.15072))     | 180K samples of LLM-generated instruction-following data     | [Github](https://github.com/superjamessyx/Generative-Foundation-AI-Assistant-for-Pathology) |
| PMC-VQA ([arXiv](https://arxiv.org/abs/2305.10415))          | 227K VQA pairs of 149K images of various modalities or diseases | [Github](https://github.com/xiaoman-zhang/PMC-VQA)           |
| PMC-OA ([arXiv](https://arxiv.org/abs/2303.07240))           | 1.6M fine-grained biomedical image-text pairs                | [Github](https://github.com/WeixiongLin/PMC-CLIP)            |
| PathCap ([arXiv](https://arxiv.org/abs/2303.07240))          | 142K pathology image-caption pairs from various sources      | [Github](https://github.com/WeixiongLin/PMC-CLIP)            |
| SwissProtCLAP ([arXiv](https://arxiv.org/abs/2302.04611))    | 441K text-protein sequence pairs                             | [Github](https://github.com/chao1224/chatdrug)               |
| MIMIC-IV ([Nature](https://www.nature.com/articles/s41597-022-01899-x)) | Clinical information for hospital stays of over 60,000 patients | [Official site](https://mimic.mit.edu/docs/iv/)              |
| MIMIC-CXR ([Nature](https://www.nature.com/articles/s41597-019-0322-0)) | 227,835 chest imaging studies with free-text reports for 65,379 patients | [PhysioNet](https://physionet.org/content/mimic-cxr/2.0.0/)  |
| TCGA                                                         | A landmark cancer genomics program, molecularly characterized over 20,000 primary cancer and matched normal samples spanning 33 cancer types | [Official site](https://www.cancer.gov/ccg/research/genome-sequencing/tcga) |

