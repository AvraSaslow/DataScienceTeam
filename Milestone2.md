## Soccer Prediction Modelling - Milestone 2 ##
*Avra Saslow
Beni Bienz*

Our project is centred around predicting the outcome of soccer matches. In our first milestone report we mentioned two datasets we were interested in: the “ultimate” database that aggregates player and match data from 11 European countries and a more detailed event stream (a list of all events in a given match coded numerically for data analysis) dataset which covers ~90% of matches from the top 5 European leagues in a 5-year period. Our overall plan was to train a classifier to predict the outcome of matches using features extracted from these two sources as input. A secondary goal was to explore assessing individual player action values, which we expect would be a natural product of searching for features that improve our predictions. Individually we wanted to learn more about different classifiers and improve our SQL skills.

Our goal for this milestone was to complete an end-to-end data pipeline, from raw input to visualization of classification output. We first spent some time familiarizing ourselves with the data, using Python’s SQLite library to query the “ultimate” database and construct pandas dataframes of selected match data. We then decided on a subset of data to use for the purposes of building the pipeline: English Premier League matches that didn’t end in a tie (allowing for binary classification). For our demo input features, we used betting odds for home team wins from several betting shops, the idea being that each set of odds represented a compressed feature set chosen by the company. We shuffled the data and split them into train and test sets, then trained a random forest classifier using sklearn. Finally, we wrote a confusion matrix function to visualize the train and test results (fig 1).


![alt text](confusionMatrix.png "Title")

*Fig 1: Example output of classifier visualized as confusion matrices. This particular run shows the train and test results of a random forest classifier trained on English Premier League games that didn’t end in ties, using home team win betting odds from various betting shops as input features.*


An unforeseen challenge we ran into was that some of the match features (e.g. shots on target) exist in the database as complex XML strings, with no helper functions provided to parse them. If we want to use these data fields moving forward, we will need to develop some method of parsing them. Thankfully, we have found user kernels on Kaggle that address this issue, such as here and here, which we can use as a starting point. The upside of this problem is that it appears the “ultimate” database has much richer detail in its match data than initially thought, which may obviate the need for the event stream dataset.

Now we have a framework for running and evaluating models, we can easily swap out features and classifier algorithms to iterate on our model. We hope to use player and match data rather than betting odds in our input features, and potentially beat the bookies with our predictions. Along the way we will gather unique statistics that provide insight into what makes specific teams and players successful. 
