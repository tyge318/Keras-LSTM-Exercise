# Langauge Modeling

Implementing a language modeling application that generates English sentence.  
Tutorial offered by [Dr. Jason Brownlee](https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/).  
First working on character level modeling, with two settings: single layer LSTM and two-layered LSTM.  
(Adding one more layer increase a lot on the training time.)  
For sure the two-layer model gives better result.    

(Update 12/12/2017) Add word level LSTM implementation.  
Here's the result:
![output](https://github.com/tyge318/Keras-LSTM-Exercise/blob/master/LanguageModeling/word-level-lstm/wordLevelLanguageModelOutput.png)

(Update 12/15/2017) Add an implementation for Microsoft Sentence Completion Challenge.  
Solving with LSTM language modeling techniques.  
The model was trained on a small subset of vocabulary (sized around 4,500), with words outside these valid words marked as 'UNK' (unknown).
Due to small vocabulary size, the performance was very bad (accuracy of 17%, worse than random guess of 1/5).  
Will try to trained with a larger vocabulary size on the next version.  
