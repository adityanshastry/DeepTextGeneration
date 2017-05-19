# DeepTextGeneration
This was submitted as a project, required to be completed as part of the coursework for Machine Learning in UMass (Fall 2016).

Project Members:
1) Aditya Narasimha Shastry - https://github.com/adityanshastry
2) Pratik Mehta - https://github.com/pratikmehta14
3) Shehzaad Dhuliawala - https://github.com/shehzaadzd

In this project, we explored various generative Deep Learning models, and tested their ability to generate discrete text data.
We used an LSTM as a baseline measure (which turned out to be a skyline measure!!), and perplexity of the generated text as a 
metric to compere the models. The dataset used for training the models was the The (20) QA bAbI tasks in the Facebook's bAbI 
dataset (https://research.fb.com/downloads/babi/). The below models were explored:
1) Generative Adversairal Networks - https://arxiv.org/pdf/1406.2661.pdf
2) Variational Auto Encoders - https://arxiv.org/pdf/1312.6114.pdf

We found that though GANs generate sentences of lower perplexity, there were a lot of repititions. In our experiments, out of 
the 500 sentences that were generated, there were only 3 unique sentences. In case of VAEs, though there was a lot of variation
in terms of the number of sentences generated, the perplexity was quite high compared to GANs. We also ran some experiments on 
some datasets with a higher vocaulary and sentence lengths. We chose the Europarl dataset (http://www.statmt.org/europarl/) for
this task. With the same model configurations for the bAbI dataset, we didnt get good results. Unfortunately, we didnt have 
enough time to tune the models for the larger dataset. 
