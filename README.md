# CS598DL4H
This project is an reimplementation of https://github.com/UzmaHasan/KCRL for CS598 Deep Learning for Healthcare 
To test, please run main.py. Update output_dir in main.py if needed.

**Repository structure**
.
└── CS598DL4H/
    ├── datasets/
    │   ├── Asia
    │   ├── LUCAS
    │   ├── Oxygen-therapy
    │   └── SACHS
    ├── model/
    │   └── actor_critic.py #[reimplemented]
    ├── reward/
    │   └── get_reward.py   #[taken from original code]
    ├── utils/
    │   ├── bic.py          #[taken from original code]
    │   ├── eval.py         #[taken from original code]
    │   └── load.py         #[reimplemented]
    └── main.py             #[reimplemented]
    
**Dataset decription**
Four datasets are provided: LUCAS, Asia, SACHS, Oxygen-therapy
Each folder has (1) a csv file from the original paper github repository, (2) a data.npy file converted from the csv file, and (3) a true_graph.npy file created from the data sources listed below:
LUCAS: http://www.causality.inf.ethz.ch/data/LUCAS.html (LUCAS0)
![image](https://user-images.githubusercontent.com/109108701/236724270-e3e0f698-5b4f-4a49-9f96-2d3fbf242c4e.png)
Asia: https://www.bnlearn.com/bnrepository/discrete-small.html#asia
![image](https://user-images.githubusercontent.com/109108701/236724367-b8ffc491-bfb9-487f-a390-0332fe5676d5.png)
SACHS: https://www.bnlearn.com/bnrepository/discrete-small.html#sachs
![image](https://user-images.githubusercontent.com/109108701/236724466-7e8732a2-5bd6-4863-94e6-aee4046efe8b.png)
Oxygen-therapy: https://arxiv.org/pdf/2010.14774.pdf
![image](https://user-images.githubusercontent.com/109108701/236724863-35b8e6d9-1f29-40eb-861d-20f47d1db09a.png)
