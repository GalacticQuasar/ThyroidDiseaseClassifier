# Machine Learning Model Comparison for Thyroid Disease Classification
This project details my process for data preprocessing, data analysis, training, optimization, and finally comparison of statistical machine learning models in the task of thyroid disease classification, given the results of a thyroid disease test.
I completed this project during my internship at [StARLinG Lab](https://starling.utdallas.edu/) at the University of Texas at Dallas, and presented my findings to Dr. Sriram Natarajan and Dr. Jey Veerasamy.

---

## **Analysis of Thyroid Disease Database Structure:**
Thyroid Disease Dataset (form UCI database): https://archive.ics.uci.edu/dataset/102/thyroid+disease

### 6 Original Databases (2800 train - 972 test, same instances but different classes, 28 attributes: 21 boolean and 7 continuous):
* allbp.[data, names, test]		- patient has high/low amt of binding protein
    * Classes: increased binding protein, decreased binding protein, negative.
* allhyper.[data, names, test]	- patient has [a type of] hyperthyroidism
    * Classes: hyperthyroid, T3 toxic, goitre, secondary toxic, negative.
* allhypo.[data, names, test]		- patient has [a type of] hypothyroidism
    * Classes: hypothyroid, primary hypothyroid, compensated hypothyroid, secondary hypothyroid, negative.
* allrep.[data, names, test]		- patient had [a type of] replacement therapy
    * Classes: replacement therapy, underreplacement, overreplacement, negative.
* dis.[data, names, test]		- patient was disagreeable
    * Classes: discordant, negative.
* sick.[data, names, test]		- patient was sick or not sick
    * Classes: sick, negative.

### Ann dataset (different instances than Ross Quinlan):
* ann-[test.data, train.data, thyroid.names]
    * 3772 learning examples
    * 3428 testing examples
    * Classes: Normal, hyperfunction, subnormal
        * Class imbalance: 92% are normal
    * 21 attributes:
        * 15 binary
        * 6 continuous

### Newly added duo datasets (3163 instances with no separate test set, 25 attributes: 18 boolean and 7 continuous):
* “hypothyroid”: .data, .names
    * Classes: hypothyroid, negative
* “sick-euthyroid”: .data, .names, .test
    * Classes: sickeuthyroid, negative

### Compilation/Archive of other databases (9172 instances with no separate test set, 28 attributes: 21 boolean and 7 continuous - same as Ross Quinlan Original 6)
* thyroid0387.[data, names]
    * Classes: &lt;all of the other datasets’ classes + some more>

### **Ross Quinlan Databases’ attributes:**
```
Attribute Name 			Possible Values
-------------- 			---------------
age: 					continuous.
sex: 					M, F.
on thyroxine: 				f, t.
query on thyroxine: 		f, t.
on antithyroid medication: 	f, t.
sick: 					f, t.
pregnant: 				f, t.
thyroid surgery: 			f, t.
I131 treatment: 			f, t.
query hypothyroid: 			f, t.
query hyperthyroid: 		f, t.
lithium: 					f, t.
goitre: 					f, t.
tumor: 					f, t.
hypopituitary: 			f, t.
psych: 					f, t.
TSH measured: 				f, t.
TSH: 					continuous.
T3 measured: 				f, t.
T3: 						continuous.
TT4 measured: 				f, t.
TT4: 					continuous.
T4U measured: 				f, t.
T4U: 					continuous.
FTI measured: 				f, t.
FTI: 					continuous.
TBG measured: 				f, t.
TBG: 					continuous.
referral source: WEST, STMW, SVHC, SVI, SVHD, other.
```

---

## **Ways to increase kNN model performance:**
* Kernel methods (uses relationship between features to get measure of closeness)
* Scaling (normalizing ranges of diff continuous attributes)
    * StandardScaler from sklearn.preprocessing
    * Convert everything 
    * Ensemble methods
        * Adaptive boosting
        * Gradient boosting (bagging)
            * Sample training data n times
            * Fit kNN model for each
            * Take average of each
        * Random forest
* [K-fold cross-validation](https://towardsdatascience.com/k-fold-cross-validation-explained-in-plain-english-659e33c0bc0)
    * Increases generalization of model to prevent overfitting training data
* Reduce dimensionality (# of attributes)
    * Remove features with low variance
    * SelectKBest function from sklearn.feature_selection
* Choose correct number of k
    * Too small → Sensitive to outliers, overfitting training data
    * Too large → High bias, underfitting training data
* Bagging
    * Randomly sample dataset into multiple subsets
        * Train algorithm on each subset
        * Combine algorithms into final model

### **Reasons to switch to decision trees instead:**
* Not ideal dataset to use kNN
    * High dimensionality (25+ features)
        * May have to use Manhattan distance to achieve 
    * kNN needs to store all instances in order to make a prediction, which means it would have to store 2800 instances

---

# Presentation (Process, Analysis, Comparison):
![Slide1](https://github.com/GalacticQuasar/ThyroidDiseaseClassifier/assets/114515524/cbd73cbd-207f-495c-97d7-83e88d412909)

## **Thyroid Disease Information**
![Slide2](https://github.com/GalacticQuasar/ThyroidDiseaseClassifier/assets/114515524/f794eb34-68c1-40b6-b4c8-d3d5835b7ee0)

First, some background information: The thyroid gland is a small butterfly-shaped organ located in the front of the neck, and its job is to regulate the release of thyroid hormones that control the rate of metabolism in the body. Thyroid disease is a general term for a medical condition that prevents the thyroid gland from correctly regulating its release of hormones. If it's releasing too much thyroid hormone, it's called *hyper*thyroidism. If it's releasing too little, it's called *hypo*thyroidism. Thyroid disease is very common, with around 20 million people in the U.S. having some type of thyroid disorder. Women, in particular, are around 5 to 8 times more likely to be diagnosed with thyroid disease than a man. People over the age of 60 are also more likely to get thyroid disease.

## **Thyroid Disease Database Overview**
![Slide3](https://github.com/GalacticQuasar/ThyroidDiseaseClassifier/assets/114515524/adeaf3e0-63c1-4401-a933-fb3f28390769)

The dataset I used was sourced from the Garvan Institute of Medical Research in Darlinghurst, Australia and the data itself was provided by Ross Quinlan (who invented the C4.5 and ID3 decision tree algorithms). The dataset had records of 9172 patients. Each instance has 28 attributes with 21 of them being binary and 7 of them continuous. Every instance is labeled with one or more of the 20 classes which you can see listed in the rightmost screenshot, with each class belonging to a larger general diagnosis.

## **Exploratory Data Analysis (Male vs Female Histogram)**
![Slide4](https://github.com/GalacticQuasar/ThyroidDiseaseClassifier/assets/114515524/19ad503e-497f-4bad-9c2d-cd411a6cfe74)

The first thing that I did after importing the dataset was to try and visualize it to see if there were any patterns. This histogram here shows the prevalence of thyroid disease in patients across different age ranges, grouped by sex. You can see that females are much more likely to have some type of thyroid disease across all age ranges. Although both males and females seem to have the highest frequency of thyroid disease around ages 60 to 75, females have another peak at ages 20 to 40. This is most likely because the thyroid plays an important role in the early development of a fetus during pregnancy, as the baby is reliant on the mother's thyroid hormone during the first 3 months of development. So it makes sense that there seems to be a higher frequency of thyroid disease in women between the ages 20 and 40.

## **Exploratory Data Analysis (Correlation Matrix)**
![Slide5](https://github.com/GalacticQuasar/ThyroidDiseaseClassifier/assets/114515524/c2cb74c8-0042-41da-bd0b-d61cca0aea9d)

The next step that I took in understanding my data was binary encoding all my boolean attributes so I could analyze them with a correlation matrix heatmap. After scanning through the heatmap, I realized that the only two features that have a significant correlation with each other are TT4 and FTI.

## **Exploratory Data Analysis (Scatterplot)**
![Slide6](https://github.com/GalacticQuasar/ThyroidDiseaseClassifier/assets/114515524/27eab92b-c117-4453-835f-1ec301719f81)

To visualize this correlation, I plotted them in a scatterplot and immediately I noticed the line of instances that show a direct correlation between TT4, which stands for Total Thyroxine, and FTI, which stands for Free Thyroxine Index. After some research, I found out that FTI is actually calculated as TT4 divided by thyroid binding capacity, so that explains why there is such a strong correlation between them in the dataset.​

## **Preprocessing (ft. Common Sense)**
![Slide7](https://github.com/GalacticQuasar/ThyroidDiseaseClassifier/assets/114515524/70c11160-b373-4484-add8-e5fc2d021a28)

I started my preprocessing by looking for obvious outliers and cleaning up my data because I was planning on using the kNN algorithm which uses distance to classify. Since I think it's pretty unlikely that these people really survived to be over 65 thousand years old, I went ahead and removed them from my dataset.

## **Preprocessing: Class Grouping (Screenshots)**
![Slide8](https://github.com/GalacticQuasar/ThyroidDiseaseClassifier/assets/114515524/f1736ceb-ec62-489f-ac44-7468663ab288)

My dataset had a total of 20 individual classifications. But, because I wanted my classifier to be simple in delivering a general diagnosis, I categorized the classes into 'normal' for someone who has no illness, 'hyperthyroidism' and 'hypothyroidism' for hyperthyroidism and hypothyroidism, respectively, 'abnormal binding protein' for too much or too little binding protein, 'non-thyroidal-sick' for those who are ill but not with a thyroid related illness, and an 'other category' so my classifier could allow for modularity with other classifiers if need be- I don't want to be misdiagnosing a patient with any of my other categories but rather, allow for it to hand off the classification task to a more specific classifier.

## **Preprocessing: Class Grouping (Pie Charts)**
![Slide9](https://github.com/GalacticQuasar/ThyroidDiseaseClassifier/assets/114515524/02b87730-7e39-4fa1-84d9-09fc9bb38200)

This is what the class grouping I did in the previous slide looks like in a pie chart. You can see the 20 classes on the left side and six on the right.

## **Preprocessing: Class Imbalance**
![Slide10](https://github.com/GalacticQuasar/ThyroidDiseaseClassifier/assets/114515524/30d9f57e-9181-418a-9fd2-3c9c8fe00cf5)

I realized that there was a big class imbalance here; the 'normal' classification being in over 70% of the dataset- so I randomly sampled the instances in the normal class to balance the classes, which I knew would be beneficial for decision trees because they are prone to class imbalance.

## **Preprocessing: Missing Values**
![Slide11](https://github.com/GalacticQuasar/ThyroidDiseaseClassifier/assets/114515524/061cd07e-9393-45c8-a8ea-e444d234a5cd)

My dataset had a lot of missing values, so I listed them here with a short description of every attribute. I ended up dropping the TBG attribute because it was not recorded for 96% of the instances in the dataset. Now I had two options for dealing with the rest of the missing data: Imputation and Deletion.

## **Preprocessing: Class Grouping**
![Slide12](https://github.com/GalacticQuasar/ThyroidDiseaseClassifier/assets/114515524/3a21c0f9-8b1d-4c3a-a3cc-6b75fc47039a)

I decided to go for deletion because it didn't change the class distribution too much and I still had enough instances remaining in each class to train my model.

## **Models and Metrics**
![Slide13](https://github.com/GalacticQuasar/ThyroidDiseaseClassifier/assets/114515524/5a5953cd-f75b-4376-8bda-e5732c8c0f1c)

The models I picked to test were k-Nearest-Neighbors, Decision Trees, Random Forest, and Gradient Tree Boosting. I know we've already gone through these models before, so I'll be concise.

kNN works by calculating the distance in n_dimensions (20 in my case) between the query point and k nearest neighbors to that point. The majority class of the nearest neighbors is the class that's predicted for the new point.

Decision trees recursively split on features for maximum information gain, eventually forming a tree of decisions that will be executed when queried to classify a new instance.

Random Forest is an ensemble method which builds multiple decision trees and combines their predictions to reduce overfitting. A final prediction is made through a vote from all the trees.

Finally, gradient tree boosting is another ensemble method which, like random forest, builds multiple decision trees but does it in a way that each decision tree corrects the errors made by the previous one.

The metrics I decided to use were K-fold cross-validation and confusion matrices. K-fold cross-validation would allow me to see more effectively how the model would perform on unseen data, and confusion matrices are useful for giving a visual representation of the model's classifications and misclassifications so I can see where it may be going wrong.

## **kNN Classifier**
![Slide14](https://github.com/GalacticQuasar/ThyroidDiseaseClassifier/assets/114515524/ee0c2d76-e68a-4540-a87d-47e9e62ee3f5)

I started off with the kNN classifier. First I normalized the scale of all my features, then I used scikit-learn's GridSearch Cross Validation to find the best-performing value of **k** nearest neighbors but, even then, it was only able to reach an average ROC AUC score of .851 and an average f1 score of .583. Looking at the confusion matrix, you can see that the kNN classifier was mis-classifying the classes of 'normal' and 'other' the most. I realized that kNN is probably not the best model for my dataset because of its high dimensionality with 20 features.

## **Decision Tree Classifier**
![Slide15](https://github.com/GalacticQuasar/ThyroidDiseaseClassifier/assets/114515524/7d009c4d-abc5-440a-9af7-42b127e2e0f6)

To address kNN's weakness I decided to use a Decision Tree Classifier. After defining the splitting criterion as entropy and finding the optimal maximum tree depth through GridSearch Cross Validation, I ended up with an average ROC AUC score of .947 and an average f1 score of .917. I think the decision tree did so much better than kNN because kNN struggles with high dimensionality and categorical data, whereas decision trees are less affected by these problems.

## **Testing and Optimization (Random Forest)**
![Slide16](https://github.com/GalacticQuasar/ThyroidDiseaseClassifier/assets/114515524/528e9000-80bb-4a22-873d-4750121fda16)

Next I wanted to implement a tree-based ensemble method, so I chose Random Forest. To find the optimal number of trees to use in my random forest classifier, I plotted the model's ROC AUC score with incremental values of n_estimators. The curve here converges around 50 trees, so that is what I used for my final tuning.

## **Random Forest Classifier**
![Slide17](https://github.com/GalacticQuasar/ThyroidDiseaseClassifier/assets/114515524/53bab3e1-415e-4b17-903b-1bd3464464b3)

The Random Forest Classifier performed by far the best, with an average ROC AUC score of .991 and an average f1 score of .928.

## **Gradient Tree Boosting Classifier**
![Slide18](https://github.com/GalacticQuasar/ThyroidDiseaseClassifier/assets/114515524/6bbd605f-de7b-4fdf-a0b2-46b0246d278c)

The Gradient Tree Boosting classifier performed almost as well as the Random Forest, with an average ROC AUC score of .989 and an average f1 score of .915.

## **Results**
![Slide19](https://github.com/GalacticQuasar/ThyroidDiseaseClassifier/assets/114515524/07ecc65d-bd40-48fd-a91b-2dfffdcb7f6e)

Ultimately, Random Forest performed the best out of all the classifiers and Gradient Boosting came in at a close second. You can see the results from my Random Forest classifier on feature importance, and it looks like because the continuous measurements had the most variance it valued them a lot more than the binary attributes. T3 and TSH seem to be the highest ranked, and that makes sense because those two values are commonly tested and looked at by doctors to determine whether a person has thyroid disease or not.
