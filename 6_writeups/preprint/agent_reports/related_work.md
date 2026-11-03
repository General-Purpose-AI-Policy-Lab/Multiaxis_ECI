# Related work: bibliography for "A multi-axis Bayesian Epoch Capabilities Index with human baselines"

Verification, 2026-10-29. Every arXiv ID was resolved through the arXiv API (export.arxiv.org): title, authors and first-submission date match the entry. Every DOI was resolved through the Crossref API. Web sources (Epoch pages, Substack, LessWrong, GitHub) were opened and checked for title, author and date. Items that could not be fully checked carry a `note = {UNVERIFIED: ...}` field and are listed in section C.

Year convention: the arXiv first-submission year, unless a journal or proceedings version was confirmed.

## A. BibTeX

```bibtex
% ===================== 1. Epoch Capabilities Index lineage =====================

@misc{ho2025rosetta,
  title         = {A Rosetta Stone for {AI} Benchmarks},
  author        = {Ho, Anson and Denain, Jean-Stanislas and Atanasov, David and Albanie, Samuel and Shah, Rohin},
  year          = {2025},
  eprint        = {2512.00193},
  archivePrefix = {arXiv}
}

@misc{epoch2026ecidocs,
  title        = {{ECI} documentation},
  author       = {{Epoch AI}},
  howpublished = {\url{https://epoch.ai/data/eci-documentation}},
  year         = {2026},
  note         = {Accessed 2026-10-29. Page lists no authors and no date (copyright 2026).}
}

@misc{epoch2026ecipublic,
  title        = {eci-public: {Epoch} Capabilities Index},
  author       = {{Epoch AI}},
  howpublished = {\url{https://github.com/epoch-research/eci-public}},
  year         = {2026},
  note         = {Repository created 2026-02-01. Companion stitching code: \url{https://github.com/epoch-research/benchmark-stitching}}
}

@misc{barry2026tires1,
  title        = {Kicking the Tires of the {Epoch} Capabilities Index ({ECI}) Part 1: Introduction and Replication},
  author       = {Barry, Alexander},
  year         = {2026},
  month        = feb,
  howpublished = {\url{https://abstatisticalconsulting.substack.com/p/kicking-the-tires-of-the-epoch-capabilities}},
  note         = {Substack, 11 February 2026}
}

@misc{barry2026tires2,
  title        = {Kicking the Tires of the {Epoch} Capabilities Index ({ECI}) Part 2: Uncertainty and Alternative Models},
  author       = {Barry, Alexander},
  year         = {2026},
  month        = feb,
  howpublished = {\url{https://abstatisticalconsulting.substack.com/p/kicking-the-tires-of-the-epoch-capabilities-741}},
  note         = {Substack, 15 February 2026}
}

@misc{barry2026replication,
  title        = {{ECI\_replication}: Code for replication and improvements of the {Epoch} Capabilities Index},
  author       = {Barry, Alexander},
  year         = {2026},
  howpublished = {\url{https://github.com/Alexander-Barry/ECI_replication}},
  note         = {R and Stan. Repository created 2026-02-04.}
}

@misc{denain2026accelerated,
  title        = {Have {AI} capabilities accelerated?},
  author       = {Denain, Jean-Stanislas and Barry, Alexander},
  year         = {2026},
  month        = apr,
  howpublished = {\url{https://epoch.ai/publications/have-ai-capabilities-accelerated}},
  note         = {Epoch AI, 16 April 2026}
}

@misc{domenech2026mapping,
  title        = {Mapping {AI} Capabilities to Human Expertise on the {Rosetta Stone} ({Epoch} Capabilities Index)},
  author       = {Domenech, Laura and Andr{\'e}oletti, J{\'e}r{\'e}my},
  year         = {2026},
  month        = mar,
  howpublished = {\url{https://www.lesswrong.com/posts/cfbdyJGbHkY8rPesE/mapping-ai-capabilities-to-human-expertise-on-the-rosetta-1}},
  note         = {LessWrong, 9 March 2026; crosspost from the General-Purpose AI Policy Lab research blog}
}

@misc{essifi2026fouraxis,
  title        = {A Four-Axis {Bayesian} {Epoch} Capabilities Index with Human Baselines},
  author       = {Essifi, Yassine and Andr{\'e}oletti, J{\'e}r{\'e}my},
  year         = {2026},
  month        = sep,
  howpublished = {\url{https://www.lesswrong.com/posts/gbJYR8TMaQKsvjrkh/a-four-axis-bayesian-epoch-capabilities-index-with-human}},
  note         = {LessWrong, 7 September 2026; crosspost from the General-Purpose AI Policy Lab research blog}
}

% ===================== 2. Latent-variable models of AI evaluation =====================

@misc{polo2024tinybenchmarks,
  title         = {{tinyBenchmarks}: evaluating {LLMs} with fewer examples},
  author        = {Maia Polo, Felipe and Weber, Lucas and Choshen, Leshem and Sun, Yuekai and Xu, Gongjun and Yurochkin, Mikhail},
  year          = {2024},
  eprint        = {2402.14992},
  archivePrefix = {arXiv}
}

@misc{polo2024multiprompt,
  title         = {Efficient multi-prompt evaluation of {LLMs}},
  author        = {Maia Polo, Felipe and Xu, Ronald and Weber, Lucas and Silva, M{\'i}rian and Bhardwaj, Onkar and Choshen, Leshem and de Oliveira, Allysson Flavio Melo and Sun, Yuekai and Yurochkin, Mikhail},
  year          = {2024},
  eprint        = {2405.17202},
  archivePrefix = {arXiv}
}

@misc{polo2024sloth,
  title         = {Sloth: scaling laws for {LLM} skills to predict multi-benchmark performance across families},
  author        = {Maia Polo, Felipe and Somerstep, Seamus and Choshen, Leshem and Sun, Yuekai and Yurochkin, Mikhail},
  year          = {2024},
  eprint        = {2412.06540},
  archivePrefix = {arXiv}
}

@misc{hofmann2025fluid,
  title         = {Fluid Language Model Benchmarking},
  author        = {Hofmann, Valentin and Heineman, David and Magnusson, Ian and Lo, Kyle and Dodge, Jesse and Sap, Maarten and Koh, Pang Wei and Wang, Chun and others},
  year          = {2025},
  eprint        = {2509.11106},
  archivePrefix = {arXiv},
  note          = {COLM 2025 (venue per the arXiv PDF header)}
}

@misc{ruan2024observational,
  title         = {Observational Scaling Laws and the Predictability of Language Model Performance},
  author        = {Ruan, Yangjun and Maddison, Chris J. and Hashimoto, Tatsunori},
  year          = {2024},
  eprint        = {2405.10938},
  archivePrefix = {arXiv}
}

@article{ilic2024evidence,
  title   = {Evidence of interrelated cognitive-like capabilities in large language models: Indications of artificial general intelligence or achievement?},
  author  = {Ili{\'c}, David and Gignac, Gilles E.},
  journal = {Intelligence},
  volume  = {106},
  pages   = {101858},
  year    = {2024},
  doi     = {10.1016/j.intell.2024.101858},
  eprint  = {2310.11616},
  archivePrefix = {arXiv}
}

@article{burnell2023rethink,
  title   = {Rethink reporting of evaluation results in {AI}},
  author  = {Burnell, Ryan and Schellaert, Wout and Burden, John and Ullman, Tomer D. and Martinez-Plumed, Fernando and Tenenbaum, Joshua B. and Rutar, Danaja and Cheke, Lucy G. and Sohl-Dickstein, Jascha and Mitchell, Melanie and Kiela, Douwe and Shanahan, Murray and Voorhees, Ellen M. and Cohn, Anthony G. and Leibo, Joel Z. and Hernandez-Orallo, Jose},
  journal = {Science},
  volume  = {380},
  number  = {6641},
  pages   = {136--138},
  year    = {2023},
  doi     = {10.1126/science.adf6369}
}

@misc{burnell2023revealing,
  title         = {Revealing the structure of language model capabilities},
  author        = {Burnell, Ryan and Hao, Han and Conway, Andrew R. A. and Hernandez-Orallo, Jose},
  year          = {2023},
  eprint        = {2306.10062},
  archivePrefix = {arXiv}
}

@misc{zhou2025generalscales,
  title         = {General Scales Unlock {AI} Evaluation with Explanatory and Predictive Power},
  author        = {Zhou, Lexin and Pacchiardi, Lorenzo and Mart{\'i}nez-Plumed, Fernando and Collins, Katherine M. and Moros-Daval, Yael and Zhang, Seraphina and Zhao, Qinlin and Huang, Yitian and others},
  year          = {2025},
  eprint        = {2503.06378},
  archivePrefix = {arXiv}
}

@misc{burden2023triangulation,
  title         = {Inferring Capabilities from Task Performance with {Bayesian} Triangulation},
  author        = {Burden, John and Voudouris, Konstantinos and Burnell, Ryan and Rutar, Danaja and Cheke, Lucy and Hern{\'a}ndez-Orallo, Jos{\'e}},
  year          = {2023},
  eprint        = {2309.11975},
  archivePrefix = {arXiv}
}

@article{martinezplumed2019irt,
  title   = {Item response theory in {AI}: Analysing machine learning classifiers at the instance level},
  author  = {Mart{\'i}nez-Plumed, Fernando and Prud{\^e}ncio, Ricardo B. C. and Mart{\'i}nez-Us{\'o}, Adolfo and Hern{\'a}ndez-Orallo, Jos{\'e}},
  journal = {Artificial Intelligence},
  volume  = {271},
  pages   = {18--42},
  year    = {2019},
  doi     = {10.1016/j.artint.2018.09.004}
}

@misc{lalor2016building,
  title         = {Building an Evaluation Scale using Item Response Theory},
  author        = {Lalor, John P. and Wu, Hao and Yu, Hong},
  year          = {2016},
  eprint        = {1605.08889},
  archivePrefix = {arXiv},
  note          = {EMNLP 2016}
}

@inproceedings{rodriguez2021evaluation,
  title     = {Evaluation Examples are not Equally Informative: How should that change {NLP} Leaderboards?},
  author    = {Rodriguez, Pedro and Barrow, Joe and Hoyle, Alexander Miserlis and Lalor, John P. and Jia, Robin and Boyd-Graber, Jordan},
  booktitle = {Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)},
  pages     = {4486--4503},
  year      = {2021},
  doi       = {10.18653/v1/2021.acl-long.346}
}

@misc{vania2021comparing,
  title         = {Comparing Test Sets with Item Response Theory},
  author        = {Vania, Clara and Htut, Phu Mon and Huang, William and Mungra, Dhara and Pang, Richard Yuanzhe and Phang, Jason and Liu, Haokun and Cho, Kyunghyun and Bowman, Samuel R.},
  year          = {2021},
  eprint        = {2106.00840},
  archivePrefix = {arXiv},
  note          = {ACL 2021}
}

@misc{zhuang2023position,
  title         = {Position: {AI} Evaluation Should Learn from How We Test Humans},
  author        = {Zhuang, Yan and Liu, Qi and Pardos, Zachary A. and Kyllonen, Patrick C. and Zu, Jiyun and Huang, Zhenya and Wang, Shijin and Chen, Enhong},
  year          = {2023},
  eprint        = {2306.10512},
  archivePrefix = {arXiv},
  note          = {First posted as "Efficiently Measuring the Cognitive Ability of LLMs: An Adaptive Testing Perspective"; current arXiv title shown}
}

@misc{kipnis2024metabench,
  title         = {metabench -- A Sparse Benchmark of Reasoning and Knowledge in Large Language Models},
  author        = {Kipnis, Alex and Voudouris, Konstantinos and Schulze Buschoff, Luca M. and Schulz, Eric},
  year          = {2024},
  eprint        = {2407.12844},
  archivePrefix = {arXiv}
}

@misc{truong2025amortized,
  title         = {Reliable and Efficient Amortized Model-based Evaluation},
  author        = {Truong, Sang and Tu, Yuheng and Liang, Percy and Li, Bo and Koyejo, Sanmi},
  year          = {2025},
  eprint        = {2503.13335},
  archivePrefix = {arXiv}
}

@misc{zhou2025lost,
  title         = {Lost in Benchmarks? Rethinking Large Language Model Benchmarking with Item Response Theory},
  author        = {Zhou, Hongli and Huang, Hui and Zhao, Ziqing and Han, Lvyuan and Wang, Huicheng and Chen, Kehai and Yang, Muyun and Bao, Wei and others},
  year          = {2025},
  eprint        = {2505.15055},
  archivePrefix = {arXiv}
}

@misc{yao2025jeirt,
  title         = {{JE-IRT}: A Geometric Lens on {LLM} Abilities through Joint Embedding Item Response Theory},
  author        = {Yao, Louie Hong and Jarvis, Nicholas and Zhan, Tiffany and Ghosh, Saptarshi and Liu, Linfeng and Jiang, Tianyu},
  year          = {2025},
  eprint        = {2509.22888},
  archivePrefix = {arXiv}
}

@misc{shang2026contextual,
  title         = {{LLM} Evaluation on Unseen Questions: Contextual Multidimensional {IRT} Model},
  author        = {Shang, Ergan and Tang, Weijing and He, Yinqiu},
  year          = {2026},
  eprint        = {2608.22295},
  archivePrefix = {arXiv}
}

@misc{fonsecarivera2026irtsafety,
  title         = {Item Response Theory for {AI} Safety},
  author        = {Fonseca Rivera, Joshua and Shah, Neil and Africa, David Demitri and Voudouris, Konstantinos},
  year          = {2026},
  eprint        = {2608.05086},
  archivePrefix = {arXiv}
}

@misc{maimon2025skills,
  title         = {From Benchmarks to Skills: Low-Rank Factors for {LLM} Evaluation},
  author        = {Maimon, Aviya and Cohen, Amir DN and Vishne, Gal and Ravfogel, Shauli and Tsarfaty, Reut},
  year          = {2025},
  eprint        = {2507.20208},
  archivePrefix = {arXiv},
  note          = {UNVERIFIED venue: a search result lists it as TACL 2026 (ACL Anthology 2026.tacl-1.75); the arXiv record was verified}
}

@misc{kearns2026construct,
  title         = {Quantifying construct validity in large language model evaluations},
  author        = {Kearns, Ryan Othniel},
  year          = {2026},
  eprint        = {2602.15532},
  archivePrefix = {arXiv}
}

@misc{haznitrama2026factor,
  title         = {Large-scale factor analysis shows machine intelligence is only partially interpretable},
  author        = {Haznitrama, Faiz Ghifari and Azizy, Afrizal Hasbi and Ardi, Faeyza Rishad},
  year          = {2026},
  eprint        = {2609.36515},
  archivePrefix = {arXiv}
}

@misc{zhu2026onecapability,
  title         = {One Capability or Many? Structural and Predictive Tests of Benchmark Validity Disagree About Economic Benchmarks for Frontier {AI}},
  author        = {Zhu, Louis Yiven},
  year          = {2026},
  eprint        = {2608.29420},
  archivePrefix = {arXiv}
}

@misc{ren2024safetywashing,
  title         = {Safetywashing: Do {AI} Safety Benchmarks Actually Measure Safety Progress?},
  author        = {Ren, Richard and Basart, Steven and Khoja, Adam and Gatti, Alice and Phan, Long and Yin, Xuwang and Mazeika, Mantas and Pan, Alexander and others},
  year          = {2024},
  eprint        = {2407.21792},
  archivePrefix = {arXiv}
}

@article{liang2023helm,
  title   = {Holistic Evaluation of Language Models},
  author  = {Liang, Percy and Bommasani, Rishi and Lee, Tony and Tsipras, Dimitris and Soylu, Dilara and Yasunaga, Michihiro and Zhang, Yian and Narayanan, Deepak and others},
  journal = {Transactions on Machine Learning Research},
  year    = {2023},
  eprint  = {2211.09110},
  archivePrefix = {arXiv}
}

@article{ott2022mapping,
  title   = {Mapping global dynamics of benchmark creation and saturation in artificial intelligence},
  author  = {Ott, Simon and Barbosa-Silva, Adriano and Blagec, Kathrin and Brauner, Jan and Samwald, Matthias},
  journal = {Nature Communications},
  volume  = {13},
  pages   = {6793},
  year    = {2022},
  doi     = {10.1038/s41467-022-34591-0},
  eprint  = {2203.04592},
  archivePrefix = {arXiv}
}

@misc{akhtar2026plateau,
  title         = {When {AI} Benchmarks Plateau: A Systematic Study of Benchmark Saturation},
  author        = {Akhtar, Mubashara and Reuel, Anka and Soni, Prajna and Ahuja, Sanchit and Ammanamanchi, Pawan Sasanka and Rawal, Ruchit and Zouhar, Vil{\'e}m and Yadav, Srishti and others},
  year          = {2026},
  eprint        = {2602.16763},
  archivePrefix = {arXiv}
}

@misc{bean2025construct,
  title         = {Measuring what Matters: Construct Validity in Large Language Model Benchmarks},
  author        = {Bean, Andrew M. and Kearns, Ryan Othniel and Romanou, Angelika and Hafner, Franziska Sofia and Mayne, Harry and Batzner, Jan and Foroutan, Negar and Schmitz, Chris and others},
  year          = {2025},
  eprint        = {2511.04703},
  archivePrefix = {arXiv}
}

@misc{reuel2024betterbench,
  title         = {{BetterBench}: Assessing {AI} Benchmarks, Uncovering Issues, and Establishing Best Practices},
  author        = {Reuel, Anka and Hardy, Amelia and Smith, Chandler and Lamparth, Max and Hardy, Malcolm and Kochenderfer, Mykel J.},
  year          = {2024},
  eprint        = {2411.12990},
  archivePrefix = {arXiv}
}

% ===================== 3. Psychometrics and Bayesian factor models =====================

@book{reckase2009mirt,
  title     = {Multidimensional Item Response Theory},
  author    = {Reckase, Mark D.},
  publisher = {Springer},
  year      = {2009},
  doi       = {10.1007/978-0-387-89976-3}
}

@incollection{birnbaum1968,
  title     = {Some latent trait models and their use in inferring an examinee's ability},
  author    = {Birnbaum, Allan},
  booktitle = {Statistical Theories of Mental Test Scores},
  editor    = {Lord, Frederic M. and Novick, Melvin R.},
  publisher = {Addison-Wesley},
  address   = {Reading, MA},
  year      = {1968}
}

@article{whitely1980multicomponent,
  title   = {Multicomponent Latent Trait Models for Ability Tests},
  author  = {Whitely, Susan E.},
  journal = {Psychometrika},
  volume  = {45},
  number  = {4},
  pages   = {479--494},
  year    = {1980},
  doi     = {10.1007/BF02293610},
  note    = {Author later published as Susan E. Embretson}
}

@article{noel2007beta,
  title   = {A Beta Item Response Model for Continuous Bounded Responses},
  author  = {Noel, Yvonnick and Dauvier, Bruno},
  journal = {Applied Psychological Measurement},
  volume  = {31},
  number  = {1},
  pages   = {47--73},
  year    = {2007},
  doi     = {10.1177/0146621605287691}
}

@article{smithson2006lemon,
  title   = {A better lemon squeezer? Maximum-likelihood regression with beta-distributed dependent variables},
  author  = {Smithson, Michael and Verkuilen, Jay},
  journal = {Psychological Methods},
  volume  = {11},
  number  = {1},
  pages   = {54--71},
  year    = {2006},
  doi     = {10.1037/1082-989X.11.1.54}
}

@article{ferrari2004beta,
  title   = {Beta Regression for Modelling Rates and Proportions},
  author  = {Ferrari, Silvia and Cribari-Neto, Francisco},
  journal = {Journal of Applied Statistics},
  volume  = {31},
  number  = {7},
  pages   = {799--815},
  year    = {2004},
  doi     = {10.1080/0266476042000214501}
}

@article{loken2010fourpl,
  title   = {Estimation of a four-parameter item response theory model},
  author  = {Loken, Eric and Rulison, Kelly L.},
  journal = {British Journal of Mathematical and Statistical Psychology},
  volume  = {63},
  number  = {3},
  pages   = {509--525},
  year    = {2010},
  doi     = {10.1348/000711009X474502}
}

@article{culpepper2016fourpl,
  title   = {Revisiting the 4-Parameter Item Response Model: {Bayesian} Estimation and Application},
  author  = {Culpepper, Steven Andrew},
  journal = {Psychometrika},
  volume  = {81},
  number  = {4},
  pages   = {1142--1163},
  year    = {2016},
  doi     = {10.1007/s11336-015-9477-6}
}

@article{gibbons1992bifactor,
  title   = {Full-Information Item Bi-Factor Analysis},
  author  = {Gibbons, Robert D. and Hedeker, Donald R.},
  journal = {Psychometrika},
  volume  = {57},
  number  = {3},
  pages   = {423--436},
  year    = {1992},
  doi     = {10.1007/BF02295430}
}

@article{reise2012bifactor,
  title   = {The Rediscovery of Bifactor Measurement Models},
  author  = {Reise, Steven P.},
  journal = {Multivariate Behavioral Research},
  volume  = {47},
  number  = {5},
  pages   = {667--696},
  year    = {2012},
  doi     = {10.1080/00273171.2012.715555}
}

@article{lopes2004factor,
  title   = {Bayesian model assessment in factor analysis},
  author  = {Lopes, Hedibert Freitas and West, Mike},
  journal = {Statistica Sinica},
  volume  = {14},
  number  = {1},
  pages   = {41--67},
  year    = {2004}
}

@article{erosheva2017reflection,
  title   = {Dealing with Reflection Invariance in {Bayesian} Factor Analysis},
  author  = {Erosheva, Elena A. and Curtis, S. McKay},
  journal = {Psychometrika},
  volume  = {82},
  number  = {2},
  pages   = {295--307},
  year    = {2017},
  doi     = {10.1007/s11336-017-9564-y}
}

@article{papastamoulis2022identifiability,
  title   = {On the identifiability of {Bayesian} factor analytic models},
  author  = {Papastamoulis, Panagiotis and Ntzoufras, Ioannis},
  journal = {Statistics and Computing},
  volume  = {32},
  pages   = {23},
  year    = {2022},
  doi     = {10.1007/s11222-022-10084-4},
  eprint  = {2004.05105},
  archivePrefix = {arXiv}
}

@article{stephens2000label,
  title   = {Dealing With Label Switching in Mixture Models},
  author  = {Stephens, Matthew},
  journal = {Journal of the Royal Statistical Society: Series B (Statistical Methodology)},
  volume  = {62},
  number  = {4},
  pages   = {795--809},
  year    = {2000},
  doi     = {10.1111/1467-9868.00265}
}

@article{jasra2005label,
  title   = {Markov Chain {Monte Carlo} Methods and the Label Switching Problem in {Bayesian} Mixture Modeling},
  author  = {Jasra, Ajay and Holmes, Chris C. and Stephens, David A.},
  journal = {Statistical Science},
  volume  = {20},
  number  = {1},
  pages   = {50--67},
  year    = {2005},
  doi     = {10.1214/088342305000000016}
}

@misc{stan_labelswitching,
  title        = {Stan User's Guide: Problematic Posteriors, Label Switching in Mixture Models},
  author       = {{Stan Development Team}},
  howpublished = {\url{https://mc-stan.org/docs/stan-users-guide/}},
  note         = {Section "Label switching in mixture models"; checked against version 2.30 (section 23.2). Cite the version you consulted.}
}

@misc{rohe2020varimax,
  title         = {Vintage Factor Analysis with Varimax Performs Statistical Inference},
  author        = {Rohe, Karl and Zeng, Muzhe},
  year          = {2020},
  eprint        = {2004.05387},
  archivePrefix = {arXiv}
}

@article{comon1994ica,
  title   = {Independent component analysis, A new concept?},
  author  = {Comon, Pierre},
  journal = {Signal Processing},
  volume  = {36},
  number  = {3},
  pages   = {287--314},
  year    = {1994},
  doi     = {10.1016/0165-1684(94)90029-9}
}

@article{azzalini1985skew,
  title   = {A class of distributions which includes the normal ones},
  author  = {Azzalini, Adelchi},
  journal = {Scandinavian Journal of Statistics},
  volume  = {12},
  number  = {2},
  pages   = {171--178},
  year    = {1985}
}

@article{bazan2006skew,
  title   = {A skew item response model},
  author  = {Baz{\'a}n, Jorge Luis and Branco, M{\'a}rcia D. and Bolfarine, Heleno},
  journal = {Bayesian Analysis},
  volume  = {1},
  number  = {4},
  pages   = {861--892},
  year    = {2006},
  doi     = {10.1214/06-BA128}
}

@article{piironen2017horseshoe,
  title   = {Sparsity information and regularization in the horseshoe and other shrinkage priors},
  author  = {Piironen, Juho and Vehtari, Aki},
  journal = {Electronic Journal of Statistics},
  volume  = {11},
  number  = {2},
  pages   = {5018--5051},
  year    = {2017},
  doi     = {10.1214/17-EJS1337SI},
  eprint  = {1707.01694},
  archivePrefix = {arXiv}
}

@article{carvalho2010horseshoe,
  title   = {The horseshoe estimator for sparse signals},
  author  = {Carvalho, Carlos M. and Polson, Nicholas G. and Scott, James G.},
  journal = {Biometrika},
  volume  = {97},
  number  = {2},
  pages   = {465--480},
  year    = {2010},
  doi     = {10.1093/biomet/asq017}
}

@misc{pymc_zerosumnormal,
  title        = {pymc.{ZeroSumNormal}},
  author       = {{PyMC Developers}},
  howpublished = {\url{https://www.pymc.io/projects/docs/en/stable/api/distributions/generated/pymc.ZeroSumNormal.html}},
  note         = {API documentation; available since PyMC 4.2}
}

@article{burkner2021irt,
  title   = {Bayesian Item Response Modeling in {R} with brms and {Stan}},
  author  = {B{\"u}rkner, Paul-Christian},
  journal = {Journal of Statistical Software},
  volume  = {100},
  number  = {5},
  year    = {2021},
  doi     = {10.18637/jss.v100.i05},
  eprint  = {1905.09501},
  archivePrefix = {arXiv}
}

@article{felsenstein1985phylogenies,
  title   = {Phylogenies and the Comparative Method},
  author  = {Felsenstein, Joseph},
  journal = {The American Naturalist},
  volume  = {125},
  number  = {1},
  pages   = {1--15},
  year    = {1985},
  doi     = {10.1086/284325}
}

% ===================== 4. Bayesian computation and checking =====================

@article{hoffman2014nuts,
  title   = {The No-{U}-Turn Sampler: Adaptively Setting Path Lengths in {Hamiltonian} {Monte Carlo}},
  author  = {Hoffman, Matthew D. and Gelman, Andrew},
  journal = {Journal of Machine Learning Research},
  volume  = {15},
  pages   = {1593--1623},
  year    = {2014},
  eprint  = {1111.4246},
  archivePrefix = {arXiv},
  note    = {JMLR volume and pages from memory, UNVERIFIED; arXiv record verified}
}

@article{carpenter2017stan,
  title   = {{Stan}: A Probabilistic Programming Language},
  author  = {Carpenter, Bob and Gelman, Andrew and Hoffman, Matthew D. and Lee, Daniel and Goodrich, Ben and Betancourt, Michael and Brubaker, Marcus and Guo, Jiqiang and Li, Peter and Riddell, Allen},
  journal = {Journal of Statistical Software},
  volume  = {76},
  number  = {1},
  year    = {2017},
  doi     = {10.18637/jss.v076.i01}
}

@article{abrilpla2023pymc,
  title   = {{PyMC}: a modern, and comprehensive probabilistic programming framework in {Python}},
  author  = {Abril-Pla, Oriol and Andreani, Virgile and Carroll, Colin and Dong, Larry and Fonnesbeck, Christopher J. and Kochurov, Maxim and Kumar, Ravin and Lao, Junpeng and others},
  journal = {PeerJ Computer Science},
  volume  = {9},
  pages   = {e1516},
  year    = {2023},
  doi     = {10.7717/peerj-cs.1516}
}

@misc{nutpie,
  title        = {nutpie: {Python} wrapper for nuts-rs},
  author       = {Seyboldt, Adrian and {PyMC Developers}},
  howpublished = {\url{https://github.com/pymc-devs/nutpie}},
  note         = {Software. Repository verified; author attribution UNVERIFIED (check the repo's citation file).}
}

@article{kumar2019arviz,
  title   = {{ArviZ} a unified library for exploratory analysis of {Bayesian} models in {Python}},
  author  = {Kumar, Ravin and Carroll, Colin and Hartikainen, Ari and Martin, Osvaldo},
  journal = {Journal of Open Source Software},
  volume  = {4},
  number  = {33},
  pages   = {1143},
  year    = {2019},
  doi     = {10.21105/joss.01143}
}

@article{vehtari2017loo,
  title   = {Practical {Bayesian} model evaluation using leave-one-out cross-validation and {WAIC}},
  author  = {Vehtari, Aki and Gelman, Andrew and Gabry, Jonah},
  journal = {Statistics and Computing},
  volume  = {27},
  number  = {5},
  pages   = {1413--1432},
  year    = {2017},
  doi     = {10.1007/s11222-016-9696-4},
  eprint  = {1507.04544},
  archivePrefix = {arXiv}
}

@article{vehtari2021rhat,
  title   = {Rank-normalization, folding, and localization: An improved {$\widehat{R}$} for assessing convergence of {MCMC}},
  author  = {Vehtari, Aki and Gelman, Andrew and Simpson, Daniel and Carpenter, Bob and B{\"u}rkner, Paul-Christian},
  journal = {Bayesian Analysis},
  year    = {2021},
  doi     = {10.1214/20-BA1221},
  eprint  = {1903.08008},
  archivePrefix = {arXiv}
}

@misc{yao2022stacking,
  title         = {Stacking for Non-mixing {Bayesian} Computations: The Curse and Blessing of Multimodal Posteriors},
  author        = {Yao, Yuling and Vehtari, Aki and Gelman, Andrew},
  year          = {2020},
  eprint        = {2006.12335},
  archivePrefix = {arXiv},
  note          = {Published in JMLR 23 (2022) per common citation; journal details UNVERIFIED}
}

@article{gelman1996ppc,
  title   = {Posterior predictive assessment of model fitness via realized discrepancies},
  author  = {Gelman, Andrew and Meng, Xiao-Li and Stern, Hal},
  journal = {Statistica Sinica},
  volume  = {6},
  number  = {4},
  pages   = {733--807},
  year    = {1996}
}

@article{gabry2019visualization,
  title   = {Visualization in {Bayesian} workflow},
  author  = {Gabry, Jonah and Simpson, Daniel and Vehtari, Aki and Betancourt, Michael and Gelman, Andrew},
  journal = {Journal of the Royal Statistical Society: Series A},
  volume  = {182},
  pages   = {389--402},
  year    = {2019},
  doi     = {10.1111/rssa.12378},
  eprint  = {1709.01449},
  archivePrefix = {arXiv}
}

@misc{gelman2020workflow,
  title         = {Bayesian Workflow},
  author        = {Gelman, Andrew and Vehtari, Aki and Simpson, Daniel and Margossian, Charles C. and Carpenter, Bob and Yao, Yuling and Kennedy, Lauren and Gabry, Jonah and B{\"u}rkner, Paul-Christian and Modr{\'a}k, Martin},
  year          = {2020},
  eprint        = {2011.01808},
  archivePrefix = {arXiv}
}

% ===================== 5. Human baselines and AI-vs-human comparison =====================

@inproceedings{kwa2025horizon,
  title         = {Measuring {AI} Ability to Complete Long Software Tasks},
  author        = {Kwa, Thomas and West, Ben and Becker, Joel and Deng, Amy and Garcia, Katharyn and Hasin, Max and Jawhar, Sami and Kinniment, Megan and others},
  booktitle     = {Advances in Neural Information Processing Systems (NeurIPS)},
  year          = {2025},
  eprint        = {2503.14499},
  archivePrefix = {arXiv}
}

@misc{rein2025hcast,
  title         = {{HCAST}: Human-Calibrated Autonomy Software Tasks},
  author        = {Rein, David and Becker, Joel and Deng, Amy and Nix, Seraphina and Canal, Chris and O'Connel, Daniel and Arnott, Pip and Bloom, Ryan and others},
  year          = {2025},
  eprint        = {2503.17354},
  archivePrefix = {arXiv}
}

@misc{legris2024harc,
  title         = {{H-ARC}: A Robust Estimate of Human Performance on the Abstraction and Reasoning Corpus Benchmark},
  author        = {LeGris, Solim and Vong, Wai Keen and Lake, Brenden M. and Gureckis, Todd M.},
  year          = {2024},
  eprint        = {2409.01374},
  archivePrefix = {arXiv}
}

@misc{chollet2019measure,
  title         = {On the Measure of Intelligence},
  author        = {Chollet, Fran{\c{c}}ois},
  year          = {2019},
  eprint        = {1911.01547},
  archivePrefix = {arXiv}
}

@misc{chollet2025arcagi2,
  title         = {{ARC-AGI-2}: A New Challenge for Frontier {AI} Reasoning Systems},
  author        = {Chollet, Fran{\c{c}}ois and Knoop, Mike and Kamradt, Gregory and Landers, Bryan and Pinkard, Henry},
  year          = {2025},
  eprint        = {2505.11831},
  archivePrefix = {arXiv}
}

@misc{rein2023gpqa,
  title         = {{GPQA}: A Graduate-Level {Google}-Proof {Q\&A} Benchmark},
  author        = {Rein, David and Hou, Betty Li and Stickland, Asa Cooper and Petty, Jackson and Pang, Richard Yuanzhe and Dirani, Julien and Michael, Julian and Bowman, Samuel R.},
  year          = {2023},
  eprint        = {2311.12022},
  archivePrefix = {arXiv}
}

@article{phan2025hle,
  title         = {Humanity's Last Exam},
  author        = {Phan, Long and Gatti, Alice and Han, Ziwen and Li, Nathaniel and Hu, Josephina and Zhang, Hugh and Zhang, Chen Bo Calvin and Shaaban, Mohamed and others},
  journal       = {Nature},
  year          = {2025},
  doi           = {10.1038/s41586-025-09962-4},
  eprint        = {2501.14249},
  archivePrefix = {arXiv},
  note          = {DOI taken from the arXiv record; Nature volume/pages not checked}
}

@misc{glazer2024frontiermath,
  title         = {{FrontierMath}: A Benchmark for Evaluating Advanced Mathematical Reasoning in {AI}},
  author        = {Glazer, Elliot and Erdil, Ege and Besiroglu, Tamay and Chicharro, Diego and Chen, Evan and Gunning, Alex and Olsson, Caroline Falkman and Denain, Jean-Stanislas and others},
  year          = {2024},
  eprint        = {2411.04872},
  archivePrefix = {arXiv}
}

@inproceedings{wei2025humanbaselines,
  title         = {Position: Human Baselines in Model Evaluations Need Rigor and Transparency (With Recommendations \& Reporting Checklist)},
  author        = {Wei, Kevin L. and others},
  booktitle     = {Proceedings of the 42nd International Conference on Machine Learning (ICML)},
  series        = {PMLR},
  volume        = {267},
  year          = {2025},
  eprint        = {2506.13776},
  archivePrefix = {arXiv},
  note          = {arXiv title: "Recommendations and Reporting Checklist for Rigorous \& Transparent Human Baselines in Model Evaluations". Full co-author list (8 co-authors) not transcribed: UNVERIFIED, take it from proceedings.mlr.press/v267/wei25s.html}
}

@misc{ho2025frontiermathhuman,
  title        = {Is {AI} already superhuman on {FrontierMath}?},
  author       = {Ho, Anson},
  year         = {2025},
  month        = may,
  howpublished = {\url{https://epoch.ai/gradient-updates/is-ai-already-superhuman-on-frontiermath}},
  note         = {Epoch AI Gradient Updates, 23 May 2025}
}

@misc{justen2025biology,
  title         = {{LLMs} Outperform Experts on Challenging Biology Benchmarks},
  author        = {Justen, Lennart},
  year          = {2025},
  eprint        = {2505.06108},
  archivePrefix = {arXiv}
}

@techreport{dev2025rand,
  title       = {Toward Comprehensive Benchmarking of the Biological Knowledge of Frontier Large Language Models},
  author      = {Dev, Sunishchal and Teague, Charles and Ellison, Grant and Brady, Kyle and others},
  institution = {RAND Corporation},
  number      = {RR-A3797-1},
  year        = {2025},
  url         = {https://www.rand.org/pubs/research_reports/RRA3797-1.html},
  note        = {Published 25 November 2025. Full author list UNVERIFIED (rand.org returned HTTP 403; first four names from the PDF cover).}
}

% ===================== 6. Trends, forecasting and frontier gaps =====================

@misc{cottier2024open,
  title        = {How far behind are open models?},
  author       = {Cottier, Ben and You, Josh and Martemianova, Natalia and Owen, David},
  year         = {2024},
  month        = nov,
  howpublished = {\url{https://epoch.ai/publications/open-models-report}},
  note         = {Epoch AI, 4 November 2024}
}

@misc{edwards2026opengap,
  title        = {Open models lag state-of-the-art closed models by 4 months},
  author       = {Edwards, Jack and Emberson, Luke},
  year         = {2026},
  month        = may,
  howpublished = {\url{https://epoch.ai/data-insights/open-closed-eci-gap}},
  note         = {Epoch AI Data Insight, 29 May 2026}
}

@misc{emberson2026uschina,
  title        = {Chinese {AI} models have lagged the {US} frontier by 7 months on average since 2023},
  author       = {Emberson, Luke},
  year         = {2026},
  month        = jan,
  howpublished = {\url{https://epoch.ai/data-insights/us-vs-china-eci}},
  note         = {Epoch AI Data Insight, 2 January 2026}
}

@inproceedings{sevilla2022compute,
  title         = {Compute Trends Across Three Eras of Machine Learning},
  author        = {Sevilla, Jaime and Heim, Lennart and Ho, Anson and Besiroglu, Tamay and Hobbhahn, Marius and Villalobos, Pablo},
  booktitle     = {2022 International Joint Conference on Neural Networks (IJCNN)},
  pages         = {1--8},
  year          = {2022},
  doi           = {10.1109/IJCNN55064.2022.9891914},
  eprint        = {2202.05924},
  archivePrefix = {arXiv}
}

@misc{erdil2022algorithmic,
  title         = {Algorithmic progress in computer vision},
  author        = {Erdil, Ege and Besiroglu, Tamay},
  year          = {2022},
  eprint        = {2212.05153},
  archivePrefix = {arXiv}
}

@misc{ho2024algorithmic,
  title         = {Algorithmic progress in language models},
  author        = {Ho, Anson and Besiroglu, Tamay and Erdil, Ege and Owen, David and Rahman, Robi and Guo, Zifan Carl and Atkinson, David and Thompson, Neil and Sevilla, Jaime},
  year          = {2024},
  eprint        = {2403.05812},
  archivePrefix = {arXiv}
}

@misc{owen2024predictable,
  title         = {How predictable is language model benchmark performance?},
  author        = {Owen, David},
  year          = {2024},
  eprint        = {2401.04757},
  archivePrefix = {arXiv}
}

@misc{pimpale2025forecasting,
  title         = {Forecasting Frontier Language Model Agent Capabilities},
  author        = {Pimpale, Govind and H{\o}jmark, Axel and Scheurer, J{\'e}r{\'e}my and Hobbhahn, Marius},
  year          = {2025},
  eprint        = {2502.15850},
  archivePrefix = {arXiv}
}

@misc{maslej2025aiindex,
  title         = {Artificial Intelligence Index Report 2025},
  author        = {Maslej, Nestor and Fattorini, Loredana and Perrault, Raymond and Gil, Yolanda and Parli, Vanessa and Kariuki, Njenga and Capstick, Emily and Reuel, Anka and others},
  year          = {2025},
  eprint        = {2504.07139},
  archivePrefix = {arXiv}
}

@misc{sajadieh2026aiindex,
  title         = {Artificial Intelligence Index Report 2026},
  author        = {Sajadieh, Sha and Fattorini, Loredana and Perrault, Raymond and Gil, Yolanda and Parli, Vanessa and Santarlasci, Lapo and Pava, Juan and Maslej, Nestor and others},
  year          = {2026},
  eprint        = {2606.15708},
  archivePrefix = {arXiv}
}

@misc{schaeffer2023mirage,
  title         = {Are Emergent Abilities of Large Language Models a Mirage?},
  author        = {Schaeffer, Rylan and Miranda, Brando and Koyejo, Sanmi},
  year          = {2023},
  eprint        = {2304.15004},
  archivePrefix = {arXiv}
}
```

## B. How each entry relates, and where it goes

### 1. Epoch Capabilities Index lineage (Introduction; Background)

- **ho2025rosetta.** The source model: one ability per model, one difficulty and one slope per benchmark, fitted by stitching overlapping benchmarks. Our K=1 case reduces to it. Cite in the Introduction and when defining the 2PL. Note the author list: Epoch AI with Google DeepMind (Albanie, Shah).
- **epoch2026ecidocs, epoch2026ecipublic.** The public ECI and its anchoring (Claude 3.5 Sonnet = 130, GPT-5 = 150), which we reuse. Cite in the scale-anchoring paragraph and the data section.
- **barry2026tires1, barry2026tires2, barry2026replication.** Barry replicates the ECI, then fits Base, Improved-Normal and Improved-Beta Bayesian variants in Stan, and finds the Beta likelihood fits best. Our Beta likelihood and noise prior follow his Improved-Beta. Cite in the Introduction (direct predecessor) and the Likelihood section. Part 3 (multidimensional extensions) was announced in Part 2 and not found online.
- **denain2026accelerated.** Epoch finds acceleration on 3 of 4 metrics, the ECI among them, driven by reasoning models. Cite in the Forecasting section as context for trend shape and for the choice of a trend model.
- **domenech2026mapping.** Our first post: human tiers on the one-axis Rosetta scale, and a Top-Performer crossover forecast. It exposed the problem the multi-axis model fixes (humans near ceiling on ARC-type tasks, near chance on GPQA-type tasks, models the opposite). Cite in the Introduction as motivation.
- **essifi2026fouraxis.** The blog version of this paper. Cite once, in the Introduction or a footnote, as a prior informal version.

### 2. Latent-variable models of AI evaluation (Related work)

- **polo2024tinybenchmarks, polo2024multiprompt.** IRT used at item level to cut the number of evaluation examples. Same model family, different goal (cost reduction, not a capability scale). Related work, "IRT for LLMs".
- **polo2024sloth.** Low-dimensional skill factors that predict benchmark scores across model families. Closest in spirit to a multi-axis capability model, but frequentist and tied to compute. Related work, "multidimensional capability".
- **hofmann2025fluid.** Adaptive item selection with IRT, and a measured notion of benchmark saturation. Related work, "IRT for LLMs" and "benchmark lifecycle".
- **ruan2024observational.** Low-rank PCA capability space across about 100 models, used to forecast benchmark scores. The closest prior multidimensional capability space. Cite in Related work and in the Forecasting section.
- **ilic2024evidence.** Factor analysis finds a dominant general factor across LLM tests. Supports a strong first axis. Cite where the first axis is interpreted.
- **burnell2023rethink.** Argues for instance-level results and against single aggregate scores. Motivation for a decomposed index. Introduction.
- **burnell2023revealing.** Factor analysis on HELM finds three broad factors (reasoning, comprehension, core language). Prior evidence for more than one axis. Related work, "multidimensional capability".
- **zhou2025generalscales.** ADeLe: rubric-based demand levels on 18 general scales place models and tasks on interpretable axes. An alternative route to multiple axes, by annotation instead of latent estimation. Related work.
- **burden2023triangulation.** Bayesian inference of latent capabilities from task performance. Methodological sibling. Related work.
- **martinezplumed2019irt.** Early IRT for machine learning classifiers at instance level. Related work, first sentence of the IRT paragraph.
- **lalor2016building, rodriguez2021evaluation, vania2021comparing.** IRT for NLP test sets and leaderboards. Related work, "IRT for NLP".
- **zhuang2023position.** Adaptive testing and psychometrics for LLM evaluation. Related work.
- **kipnis2024metabench.** IRT plus factor analysis on Open LLM Leaderboard items, finding a general factor. Related work.
- **truong2025amortized.** IRT across HELM with amortized difficulty. Related work, "IRT for LLMs".
- **zhou2025lost.** IRT critique of mainstream LLM benchmarks. Related work.
- **yao2025jeirt, shang2026contextual.** Multidimensional IRT for LLMs at item level (embedding geometry; contextual MIRT for unseen questions). Closest 2025-2026 MIRT work; they model items, we model benchmark aggregates over time with human tiers. Related work, "multidimensional capability".
- **fonsecarivera2026irtsafety.** IRT applied to AI safety evaluation. Related work, short mention.
- **maimon2025skills, haznitrama2026factor, zhu2026onecapability, kearns2026construct.** 2025-2026 factor-analytic studies of benchmark matrices: low-rank skill factors, a partly interpretable general factor, a structural-vs-predictive dimensionality test, and construct validity. Use them to place our K=4 choice and the multimodality result. Related work and Discussion.
- **ren2024safetywashing.** Many safety benchmarks load on a general capabilities component. Relevant to which benchmarks load on the first axis. Discussion.
- **liang2023helm.** Multi-metric holistic evaluation; data source for burnell2023revealing. Related work, one line.
- **ott2022mapping, akhtar2026plateau.** Benchmark creation and saturation dynamics. Motivate stitching, chance floors and estimated ceilings. Related work, "benchmark lifecycle", and the ceiling paragraph of Methods.
- **bean2025construct, reuel2024betterbench.** Construct validity and benchmark quality. Discussion of what an axis measures.

### 3. Psychometrics and Bayesian factor models (Methods)

- **reckase2009mirt.** Standard MIRT reference; defines the compensatory model we fit. Methods, model definition.
- **birnbaum1968.** Origin of the 2PL and 3PL. Methods, model definition; the chance floor is a fixed-c 3PL.
- **whitely1980multicomponent.** Non-compensatory (conjunctive) multicomponent model. Cite where we report that the conjunctive link fits worse.
- **noel2007beta, smithson2006lemon, ferrari2004beta.** Beta IRT and Beta regression for bounded continuous scores; the boundary-handling literature for exact 0 and 1. Methods, likelihood.
- **loken2010fourpl, culpepper2016fourpl.** Four-parameter IRT with an upper asymptote estimated in a Bayesian setting. Methods, estimated ceilings.
- **gibbons1992bifactor, reise2012bifactor.** Bifactor structure. Cite if the bifactor loading prior is reported as a sensitivity run.
- **lopes2004factor, erosheva2017reflection, papastamoulis2022identifiability.** Rotational, sign and permutation non-identification in Bayesian factor models. Methods, identification; Results, mode-aware reporting.
- **stephens2000label, jasra2005label, stan_labelswitching.** Label switching and why only invariant quantities are interpretable. Justifies judging convergence on identified quantities (eta, D, sigma). Methods, convergence.
- **rohe2020varimax, comon1994ica.** Non-Gaussian latent factors identify the rotation (ICA, leptokurtic varimax). Cite where skew-normal abilities are argued to help pin the frame.
- **azzalini1985skew, bazan2006skew.** Skew-normal family, and a skew-normal IRT model. Methods, ability prior.
- **piironen2017horseshoe, carvalho2010horseshoe.** Horseshoe and regularized horseshoe. Cite only if the bifactor sensitivity prior is reported.
- **pymc_zerosumnormal.** Sum-to-zero ability prior that pins the location. Methods, ability prior.
- **burkner2021irt.** Bayesian IRT in Stan via brms; reference implementation for Bayesian IRT. Methods, one line.
- **felsenstein1985phylogenies.** Brownian motion along a tree as a prior on trait evolution. The lineage prior is the same construction on vendor release trees indexed by time. Methods, lineage prior.

### 4. Bayesian computation and checking (Methods; Appendix)

- **hoffman2014nuts, carpenter2017stan, abrilpla2023pymc, nutpie, kumar2019arviz.** Sampler and software stack. Methods, computation.
- **vehtari2017loo.** PSIS-LOO for model comparison across K and priors. Results, model comparison.
- **vehtari2021rhat.** Rank-normalized R-hat and bulk ESS used in the convergence readout. Methods, convergence.
- **yao2022stacking.** Non-mixing chains and multimodal posteriors; chain stacking as an alternative to pooling. Results, mode-aware reporting.
- **gelman1996ppc, gabry2019visualization, gelman2020workflow.** Posterior predictive checks, PIT plots, and the overall workflow. Methods, model checking; Appendix.

### 5. Human baselines (Methods, human tiers; Results, crossover dates)

- **kwa2025horizon, rein2025hcast.** METR time horizons, calibrated with human task times. The other main human-anchored capability scale. Related work and Discussion.
- **legris2024harc, chollet2019measure, chollet2025arcagi2.** ARC human performance data and the ARC-AGI design. Source of the human advantage on abstract reasoning that a single axis cannot fit. Data section, human tiers.
- **rein2023gpqa, phan2025hle, glazer2024frontiermath.** Expert-written benchmarks with expert or non-expert human baselines. Data section, human tier sources.
- **wei2025humanbaselines.** Review of 115 human baselines; most lack rigor. Discussion, limitations of human tiers.
- **ho2025frontiermathhuman.** Epoch's FrontierMath human contest (about 40 participants in 8 teams). Data section, human tiers; an example of team versus individual baselines.
- **justen2025biology, dev2025rand.** Biology benchmarks with PhD-level human baselines; the RAND report is also a data source. Data section.

### 6. Trends, forecasting and frontier gaps (Results; Discussion)

- **cottier2024open, edwards2026opengap.** Open-weight lag: about one year on benchmarks in 2024, 4 months on the ECI in 2026. Direct comparison for our open-vs-closed gap. Results, frontier gaps.
- **emberson2026uschina.** US-vs-China lag of 7 months on average on the ECI since 2023 (range 4-14). Direct comparison for our US-vs-China gap. Results, frontier gaps.
- **sajadieh2026aiindex, maslej2025aiindex.** AI Index reports; the 2026 edition is reported to discuss the closing US-China performance gap (secondary coverage, not checked in the PDF). Results, frontier gaps, one line.
- **sevilla2022compute, ho2024algorithmic, erdil2022algorithmic.** Compute and algorithmic-progress trends. Context for the time trend. Forecasting section, one line each.
- **owen2024predictable, pimpale2025forecasting, ruan2024observational.** Prior benchmark forecasts (compute-based, ECI-like two-step, and observational). Direct comparison for our 2030 forecasts. Forecasting section.
- **schaeffer2023mirage.** Apparent jumps can come from nonlinear metrics; a logistic link on a latent scale gives smooth ability growth. Forecasting section, one line.

## C. Items not fully verified

1. **The Barry repository named in the brief, `AlexanderTBarry/Epoch-Capabilities-Index`, returns 404** (WebFetch and GitHub API). The repository his Part 1 post links is `Alexander-Barry/ECI_replication` (R, created 2026-02-04). It is cited above. Fix the link in the project's CLAUDE.md as well.
2. **Barry Part 3** (multidimensional extensions) was announced in Part 2 and not found online. Not cited.
3. **hoffman2014nuts**: JMLR volume 15, pages 1593-1623 are from memory. The arXiv record was verified.
4. **yao2022stacking**: the JMLR 2022 venue is from memory. The arXiv record was verified.
5. **maimon2025skills**: the TACL 2026 venue comes from a search result only.
6. **wei2025humanbaselines**: ICML 2025 / PMLR 267 and arXiv 2506.13776 were verified. The co-author list was not transcribed.
7. **dev2025rand**: title, report number and date were verified. Authors after the fourth were not, because rand.org returned HTTP 403.
8. **nutpie**: the repository was verified. Author attribution (Adrian Seyboldt) was not.
9. **phan2025hle**: the Nature DOI comes from the arXiv record. Volume and pages were not checked.
10. **epoch2026ecidocs**: the page shows no authors and no date.
11. **No arXiv or DOI exists** for azzalini1985skew, lopes2004factor, gelman1996ppc or birnbaum1968. Each was confirmed by journal contents or library records.

Count: 99 BibTeX entries. 93 are fully verified; 6 carry a partial UNVERIFIED note (items 3-8 in this list).
Suggested cuts to reach about 60: carvalho2010horseshoe, gibbons1992bifactor, reise2012bifactor, piironen2017horseshoe (only if the bifactor run is dropped); comon1994ica; jasra2005label; gabry2019visualization; liang2023helm; schaeffer2023mirage; maslej2025aiindex; erdil2022algorithmic; justen2025biology; fonsecarivera2026irtsafety; haznitrama2026factor; zhu2026onecapability; lalor2016building; vania2021comparing; zhou2025lost; kearns2026construct; reuel2024betterbench.
