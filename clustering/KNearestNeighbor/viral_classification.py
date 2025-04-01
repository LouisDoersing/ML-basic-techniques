import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

all_tweets = pd.read_json("random_tweets.json", lines=True)
print(all_tweets.columns)


"""
Unfortunatly, there is no feature "Viral" that we canuse as labels, so we have to define, if a tweet is Viral or not. We
can do that, by looking at the "retweets". If the Retweet numer is high, then the tweet is Viral
"""
is_viral_lst = []
median_retweets = all_tweets['retweet_count'].median()
for t in range(len(all_tweets)):
    if all_tweets.iloc[t]['retweet_count'] >= median_retweets:
        is_viral_lst.append(1)
    else:
        is_viral_lst.append(0)

#Add "virsl" column to dataframe
all_tweets['viral'] = is_viral_lst
target = pd.DataFrame(is_viral_lst)



"""
For this classification we are using 3 features: "Tweet_length", "friends_count" and "follower_count".
"""

tweet_length = []
friends_count = []
follower_count = []
for t in range(len(all_tweets)):
    tweet_text = all_tweets.iloc[t].get("full_text", all_tweets.iloc[t].get("text", ""))
    tweet_length.append(len(tweet_text))
    friends_count.append(all_tweets.iloc[t]["user"]["friends_count"])
    follower_count.append(all_tweets.iloc[t]["user"]["followers_count"])
features_df = pd.DataFrame()
features_df['tweet_length'] = tweet_length
features_df['friends_count'] = friends_count
features_df['follower_count'] = follower_count

#merge into one dataframe
data = pd.DataFrame(features_df)
data["target"] = target



"""
Normalize the data with Scaler. Split data into Test and Training and Initialize KNN.
Visualize the relation between "n_neighbors" and score
"""

scaled_data = scale(data, axis=0)

x_train, x_test, y_train, y_test = train_test_split(features_df, target, test_size=0.2, random_state=1)
y_train = y_train.iloc[:, 0]
y_test = y_test.iloc[:, 0]


scores = []
for i in range(1, 200):
    c = KNeighborsClassifier(n_neighbors=i)
    c.fit(x_train, y_train)
    scores.append(c.score(x_test, y_test))

#Visualize
print(scores)
plt.plot(scores)
plt.show()

