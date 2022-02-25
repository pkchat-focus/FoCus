# Call for Customized Conversation: Customized Conversation Grounding Persona and Knowledge

Source codes for the baseline models of **[Call for Customized Conversation: Customized Conversation Grounding Persona and Knowledge](https://arxiv.org/abs/2112.08619)**, accepted at AAAI-22.



### Environment Setting
We trained the models under the setting of torch==1.5.0, transformers==4.5.0, tensorboardX==2.1, etc.

We exploited [pytorch-ignite](https://github.com/pytorch/ignite) from pytorch.

Please see **requirements.txt** file for more information.


### Dataset [**[FoCus dataset v2](https://drive.google.com/file/d/1bHqYj-tWgd0i1Wnst-bJ30lYrmRwTPd-/view?usp=sharing)**]
This data is the modified version of the original data (which is reported in the paper) after ethical inspection.

We put train, valid, test files of the dataset in the **data** folder. (The test set will be available after March 2022.)

You should create directories named **infer_log_focus, train_log_focus, test_log_focus, models** under FoCus folder.

The project directory should follow this directory structure:

    📦FoCus
    ┣ 📂data
    ┃ ┗ 📜train.json
    ┃ ┗ 📜valid.json
    ┣ 📂ignite
    ┣ 📂infer_log_focus
    ┣ 📂models
    ┣ 📂python_tf_idf
    ┣ 📂test_log_focus
    ┣ 📂train_log_focus
    ┣ 📜classification_modules.py
    ┣ 📜data_utils.py
    ┣ 📜evaluate_test.py
    ┣ 📜evaluate_test_ppl.py
    ┣ 📜inference.sh
    ┣ 📜inference_test.py
    ┣ 📜LICENSE
    ┣ 📜README.md
    ┣ 📜requirements.txt
    ┣ 📜test.sh
    ┣ 📜train.sh
    ┣ 📜train_focus.py
    ┗ 📜utils_focus


### Training the models
Uncomment the command lines in the **train.sh** file, to start training the model. 

    sh train.sh 


### Evaluation
Uncomment the command lines in the **test.sh** file, to evaluate the model on the test set. 

    sh test.sh


### Inference
Uncomment the command lines in the **inference.sh** file, to generate utterances with the trained models.

    sh inference.sh


### Join Our Workshop @ [COLING 2022](https://coling2022.org/)
We are going to hold **[the 1st workshop on Customized Chat Grounding Persona and Knowledge](https://sites.google.com/view/persona-knowledge-workshop)** in Octorber 2022.
Stay tuned for our latest updates!

Written by [Yoonna Jang](https://github.com/YOONNAJANG).


(c) 2021 [NCSOFT Corporation](https://kr.ncsoft.com/en/index.do) & [Korea University](http://blp.korea.ac.kr/). All rights reserved.