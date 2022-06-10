Deep Neural Networks— Personality Classification based on Text
==============================================================

This summer, I’ve been spending a lot of time on learning more about neural networks. I’ve taken Intro to ML in CMU, yes, but learning through lecture notes is one thing, learning through doing is another. I much prefer the second option — deeply understanding the material, then implementing it in code to see how it works.

After completing the introductory courses of DeepLearning.ai, I’ve decided to put my skills to test with training a neural network that can, based on text, classify your personality.

![](https://miro.medium.com/max/1400/1*AaUBpDoCItK4lzk-dyZEug.png)

The Dataset
===========

Neural Networks are a type of supervised learning, meaning that for us to be able to classify text with personality, we need a training set of labelled examples.

This is a challenging dataset to find, but thankfully, us humans love to join communities that we identify with. Given that the 16 personalities test has been done millions of times, it is no surprise that there are reddit subreddits for each of the 16 personality types, filled with questions and writings of people of that personality.

Perfect. We have our dataset, but first, we have to figure out a way to take the posts from reddit, to a file which we can read. We have to have a large number of examples for our Deep Neural Network to train well on, so this is a task that we cannot do manually, we have to automate it.

Steps of our Implementation
===========================

Parsing Reddit
--------------

This was the difficult one. Learning how to interact with APIs are always an interesting journey, because you have to read and understand the documentation, and deal with requests.

Thankfully, there is a library in place, Python Reddit API Wrapper (PRAW), that makes parsing reddit as easy as a module call.

*   First, we have to go to [https://www.reddit.com/prefs/apps](https://www.reddit.com/prefs/apps) to create a new application. For the type of the application we select a script. This interface creates clientID, clientSecretKey, userAgent variables that we will use to authenticate our API requests.
*   While creating your app, pay attention to how you set your userAgent variable. The name of your application needs to fit a certain template, or else, Reddit will not authenticate you. Try to avoid keywords like “bot”, “auto”, “scraper” in your descriptions as well. So if you are seeing 403 HTTP errors, this is probably why. The recommended format is `<platform>:<app ID>:<version string> (by u/<Reddit username>).`
*   At this point, we can give these credentials to create our Reddit instance with our PRAW library. Currently, our access will be limited to read only, which is perfect for this project. If we wish to go beyond this, we would need to also provide our reddit username and password.

With the PRAW reddit instance in place, the rest of the algorithm is relatively simple. We iterate through all of the subreddits, and request the top posts from each subreddit.

We then take these posts, get their title and their body, and store it. After doing this operation for each post in each subreddit, we return both an array of the texts we have collected, and another array of the same length, where each entry is subreddit the post was taken from (essentially its label).

Notice that we have a sleep timer in the getPostsFromSubredditList function, this is because without it, due to the sheer number of requests we make, we would get a 503 HTTP response (overload).

Pre-Processing our Text
-----------------------

Our model is going to be trained on this data, so it is paramount that we get rid of words that are either going to disturb how our model learns or provides no information.

1.  For starters, if personality is mentioned in text (“entj, infj”), we want to get rid of them, because we want our model to learn how to analyze patterns, not how to search for any MBTI personality in text.
2.  It is also a good practice to get rid of words with less than 3 characters, any punctuations, any whitespace, make all uppercase letters lowercase, and get rid of the stop-words (‘me, you, them…’).

The operations here are not complex, but still, even simple operations like this make us try to find better ways of achieving our tasks. The two new functions that I have learned were:

```
str.makeTrans() & translate()
```

and they are really convenient methods that you can read more about, [here](https://www.delftstack.com/howto/python/python-replace-multiple-characters/#use-str.replace-to-replace-multiple-characters-in-python).

Tokenization (…we are nearly there)
-----------------------------------

Tokenization is at the heart of how we are able to translate our texts (which are undetermined length) into a tensor that our Tensorflow models can run.

We will be using a bag of words method, which works as follows:

1.  First, we have to decide how many words we should keep tracking, name this variable _num\_words_. I’ve set this value as 600.
2.  Then, we place all of the words from all of our training text into a bag and we select the most common _num\_words_ (600).
3.  Now our bag acts like a dictionary. It assigns a unique index to each of our top 600 words. Later, when we give it a text, it will create a vector of 600 elements, storing how frequently each word has been seen in text.

See below for a visual representation:

![](https://miro.medium.com/max/1400/0*DmxZgoe8VemlkQmd.jpeg)Image representing the bag of words method

Models (finally!)
-----------------

We’ll use the Keras framework on top of Tensorflow to create, train, and evaluate our Deep Neural Network. There are couple of things to pay attention to:

1.  Our network is going to be a **_deep_** NN, meaning, it will have more than one hidden layers. In our example, we have 4 hidden layers and 1 output layer.
2.  We will be using **_‘relu’_** activation for all of the hidden layers, and **_‘softmax’_** activation for the final output layer.
3.  The **_softmax_** activation for the final layer allows us to normalize the output vector and represent it as a probability distribution over predicted output classes.
4.  For our loss function, we want to choose a function that supports **one-hot encoding** for our labels, and hence we Sparse Categorical Cross entropy function.

Accuracy of Model
=================

When we analyze the accuracy of the model, with our test set, we get the following performance:

```
accuracy = 0.1185
```

When looking at accuracy for each of the 4 categories (Extraverted vs Introverted, Thinking vs Feeling, …), we get the following performance:

```
 E/I         S/N        T/F        J/P  
\[0.54589923 0.63688244 0.51785423 0.58356432\]
```

We see small improvements here and there, but nothing substantial. However, we need to put some things into perspective here. The Native Bayes accuracy for this problem is

```
nativeBayesAccuracy = 1/16 = 0.0625
```

Meaning, that our model is nearly twice as better as randomly guessing.

Where can we go from here?
--------------------------

1.  The Reddit subreddits are filled with memes, not a lot of text. So, while I tried to pull tens of thousands of posts, of those, only 6800 were text posts. This is not enough to train our model well.
2.  Reddit subreddits are probably not representative of how people make posts in general. It would be way more helpful if we could have a sample of writing from people we know to be a certain personality.

Both of these points can be resolved by finding a better dataset, like [this Kaggle dataset](https://www.kaggle.com/datasets/datasnaek/mbti-type?resource=download). Next steps would be to try to train the model on Kaggle dataset.

Funny Observations
==================

**Observation 1:** If the text includes profanity, the network classifies the person as ‘Extraverted’. Take a text from any introverted subreddit, add 8–9 ‘fuck’s at the end of the post, and **voilà,** in the eyes of the network, you have an extraverted individual.

**Observation 2:** For the (Thinking vs Feeling) classification, the model is very sensitive to the word ‘love’. It is not sensitive to ‘laugh’, or ‘cry’, or ‘emotions’, but add a couple of ‘love’s at the end of the post, and **voilà,** in the eyes of the network, you have a very emotional individual.

Code on Github
==============

If you wish to experiment with this code on your own, feel free to check out [here](https://github.com/mdbirlikci/DNN_personality_classification).
