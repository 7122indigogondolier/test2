"""
Author: Utkrist P. Thapa '21
        Abhi Jha '21
        Tina Jin '21
Program File: Implementing logistic regression to predict admission
Test 1
"""

import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class LogisticRegression():
    def __init__(self):
        self.weights = tf.Variable(tf.random.normal([12, 2]))
        self.bias = tf.Variable(tf.random.normal([2]))

    def predict(self, x):
        prediction = tf.add(tf.matmul(x, self.weights), self.bias)
        return prediction

    def cross_entropy(self, y, y_pred):
        y_values = tf.cast(y, dtype=tf.int32)
        y_values = tf.one_hot(y_values, 2)
        loss = tf.nn.softmax_cross_entropy_with_logits(y_values, y_pred)
        return tf.reduce_mean(loss)

    def train(self, batch_x, batch_y, lr):
        optimizer = tf.optimizers.SGD(learning_rate=lr)
        with tf.GradientTape() as g:
            y_pred = self.predict(batch_x)
            loss = self.cross_entropy(batch_y, y_pred)
            grads = g.gradient(loss, [self.weights, self.bias])
            optimizer.apply_gradients(zip(grads, [self.weights, self.bias]))
        return [loss, y_pred]

    def accuracy_calc(self, y, y_pred):
        y = tf.cast(y, dtype=tf.int32)
        predictions = tf.cast(tf.argmax(y_pred, axis=1), dtype=tf.int32)
        predictions = tf.cast(tf.equal(y, predictions), dtype=tf.float32)
        return tf.reduce_mean(predictions)

    def getWeights(self):
        return self.weights

    def getBias(self):
        return self.bias

def plot_loss(lyst):
    plt.title("Loss over iterations for Logistic Regression Model")
    x = range(len(lyst))
    y = lyst
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.plot(x, y, 'r')
    plt.show()

def preprocess(datafile):
    df = pd.read_csv(datafile, header=0)
    colnames  = ['SN', 'GRE', 'TOEFL', 'URATE', 'SOP',
        'LOR', 'CGPA', 'RES', 'ADM', 'RACE', 'SES']
    reorder_colnames = ['SN', 'GRE', 'TOEFL', 'URATE', 'SOP',
        'LOR', 'CGPA', 'RES', 'SES', 'ADM', 'RACE']

    df.columns = colnames
    df.dropna(inplace=True)
    df = df.reindex(columns=reorder_colnames)

    df = pd.get_dummies(df, columns=['RACE'])
    outputlast = ['SN', 'GRE', 'TOEFL', 'URATE', 'SOP',
        'LOR', 'CGPA', 'RES', 'SES', 'RACE_Asian', 'RACE_african american', 'RACE_latinx',
        'RACE_white', 'ADM']
    df = df.reindex(columns=outputlast)
    return df

def main():
    batches = 300
    learning_rate = 0.05
    batch_size = 60
    datafile = "Admission_Predict.csv"

    df = preprocess(datafile)
    features = ['AGE', 'ANE', 'CR_P', 'DIA', 'EJ_F', 'HI_BL','PLA',
            'SE_CR', 'SE_SO', 'SEX', 'SMOK', 'TIME']
    features = ['GRE', 'TOEFL', 'URATE', 'SOP', 'LOR', 'CGPA',
        'RES', 'SES', 'RACE_Asian', 'RACE_african american', 'RACE_latinx',
        'RACE_white']
    target = 'ADM'

    x = df[features]
    source_y = []
    for item in df[target]:
        if item < 0.73:
            source_y.append(0)
        else:
            source_y.append(1)
    y = pd.DataFrame(source_y, columns=[target])
    y = y[target]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
    sc = StandardScaler()
    sc.fit(x_train)
    x = tf.cast(sc.transform(x_train), dtype=tf.float32)
    y = tf.cast(y_train, dtype=tf.float32)

    sc.fit(x_test)
    x_test = tf.cast(sc.transform(x_test), dtype=tf.float32)


    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.repeat().shuffle(len(x)).batch(batch_size)

    lgr_model = LogisticRegression()
    accuracy_lyst = []
    loss_lyst = []

    for batch_number, (batch_x, batch_y) in enumerate(dataset.take(batches), 1):
        [loss, y_pred] = lgr_model.train(batch_x, batch_y, learning_rate)
        accuracy = lgr_model.accuracy_calc(batch_y, y_pred)
        accuracy_lyst.append(accuracy)
        loss_lyst.append(loss)
        print("Training Batch number: %i, loss: %f, accuracy:%f" % (batch_number, loss, accuracy))

    print("Average accuracy: ", tf.reduce_mean(accuracy_lyst).numpy())
    print("Final loss: ", loss_lyst[-1].numpy())
    print("")

    #plot_loss(loss_lyst)

    print("-----Testing with unseen test data (20% of the dataset)-----")
    predictions = lgr_model.predict(x_test)
    accuracy = lgr_model.accuracy_calc(y_test, predictions)
    print("Test accuracy: ", accuracy.numpy())
    print("")
    print("-----")
    print("Utkrist P. Thapa '21")
    print("Abhi Jha '21")
    print("Tina Jin '21")
    print("Washington and Lee University")
    print("")

if __name__=="__main__":
    main()
