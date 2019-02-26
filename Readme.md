# Readme
## Datasets
In this work we use three different datasets:
### Waseem and Hovy dataset
This dataset is describe in Waseem and Hovy [1] and can be download at http://github.com/zeerakw/hatespeech. 
### Davidson et al. dataset
This 
### SemEval dataset
## Experiments Description and Instructions to run
### Experiment 1
We reproduced the Agrawal and Awekar [3] and Badjatiya et al. [2] best reported models, following closely their paper description and the companion code.
To run the Experiment for one and other method you can run:
```
$python Experiment_1.py
```
### Experiment 2
In this experiment we take into account the issues we observed in the original implementation and modified the code consequently.
#### Agrawal and Awekar model
We re-conducted the same method proposed by Agrawal and Awekar [1] but this time making first the train-test splitting and then oversamplingthe train set before training the models. 
#### Badjatiya et al model
We re-run thesame method proposed by Badjatiya et al. [2] but this timeextracting features only from the set train(by using theLSTM-based architecture), then training the GBDT classifierwith these features over the same set풯train, and reportingall the metrics over풯test.
### Experiment 3
To estimate how well do these models generalize to other dataset from the same domain, evaluate those models –generated on the complete Waseem & Hovy dataset on the SemEval2019 dataset.
To run the Experiment for one and other method you can run:
```
$python Experiment_3.py
```
### Experiment 4
We partitioned the Wassem & Hovy dataset into trainning and testing sets, ensuring that no user is repeated between train and test set, and also ensuring at least an 85% of tweets of each class are in the train set. 
To run the Experiment for one and other method you can run:
```
$python Experiment_3.py
```
### Experiment 5
In this experiment we perform a 10-fold cross validation considering partitions with no overlapping users between the train and test sets using the enriched dataset.

### Experiment 6
To corroborate the generalization of the resulting model we use our newly created dataset to train the modelsproposed by Badjatiya et al. [2] and by Agrawal and Awekar[1]. Then we evaluate these models on previously unseen databy classifying tweets in the SemEval 2019 set.

### References

[1] Z. Waseem, D. Hovy _Hateful Symbols or Hateful People\? Predictive Features for Hate Speech on Detection on Twitter_

```                    
@inproceedings{waseem2016hateful,
  title={Hateful symbols or hateful people? predictive features for hate speech detection on twitter},
  author={Waseem, Zeerak and Hovy, Dirk},
  booktitle={Proceedings of the NAACL student research workshop},
  pages={88--93},
  year={2016}
}
```
[2]  P. Badjatiya, S. Gupta, M. Gupta, V. Varma _Deep learning for hate speech detection in tweets_
```
@inproceedings{badjatiya2017deep,
  title={Deep learning for hate speech detection in tweets},
  author={Badjatiya, Pinkesh and Gupta, Shashank and Gupta, Manish and Varma, Vasudeva},
  booktitle={Proceedings of the 26th International Conference on World Wide Web Companion},
  pages={759--760},
  year={2017},
  organization={International World Wide Web Conferences Steering Committee}
}
```
[3] S. Agrawal, A. Awekar _Deep learning for detecting cyberbullying across multiple social media platforms_
```
@inproceedings{agrawal2018deep,
  title={Deep learning for detecting cyberbullying across multiple social media platforms},
  author={Agrawal, Sweta and Awekar, Amit},
  booktitle={European Conference on Information Retrieval},
  pages={141--153},
  year={2018},
  organization={Springer}
}
```