## Soccer Game Outcome & Player Value Prediction Modelling
**Team:** *Avra Saslow,
Beni Bienz*

**Summary:**
Data science in soccer is a relatively new field of study compared with other sports, as the sparsity of scoring events and lack of explicit numerical data has prevented researchers from developing techniques that can be usefully leveraged. However, in recent years, the common perception that nothing can be learned from the data is changing, as more complex systems for analysis are developed by companies such as [Opta](https://www.optasports.com/) and top tier soccer teams begin to hire mathematicians and data scientists. Notably, [Liverpool F.C.](https://www.nytimes.com/2019/05/22/magazine/soccer-data-liverpool.html) have been enjoying immense success since hiring a Cambridge Physics PhD in 2015 to head up their research team, culminating in this year’s Champions League win. Nevertheless, soccer data analytics is still very much in its Wild West phase compared with other sports, which makes us keen to explore it!

**Datasets:**
We came across two datasets of interest. The first is a [comprehensive database](https://www.kaggle.com/hugomathien/soccer) of matches and player information, including match results, player attributes and betting odds for the top European leagues from 2008 - 2016. The second consists of [event stream](https://www.kaggle.com/secareanualin/football-events) data, which is a numerical timeline of player actions, goals and other notable events during a game. Using these datasets, we are interested primarily in predicting match outcomes, and secondarily assessing individual player performance.

**Techniques:**
We plan to explore this data by utilizing a range of classifiers such as `SVMs`, `Logistic Regression`, `CatBoost` and any others we find interesting. We will then seek to add features to the input data, which we expect will reveal some interesting statistics about tactics, player habits, formations, etc.

**Goals:**
Bookies (the people that set the odds) successfully predict game outcome 53% of the time -  we’d like to equal or beat this number. We are also excited about discovering soccer insights that go beyond the statistics which are traditionally reported (possession, shots on target, pass accuracy etc.). We both hope to learn more about the strengths and weaknesses of different classifiers, and get some decent practice querying a large dataset using `SQLite`.