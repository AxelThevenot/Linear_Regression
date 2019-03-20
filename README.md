## Linear regression principle

The linear regression is one the basic way to forge a rapport between variables. As its name indicates, the principle is to reduce a function to a linear form. In other words, the linear regression seeks to model associations between variables. Like other regression models, the linear regression model is used both to predict a phenomenon and to explain it.

The linear regression can also be use for classification cases. It works with the same method but the cost function will be calculate with a different manner. 

![linear regression by gradient descent](linear_regression_gradient.gif)


## Linear regression application

### Least squares method 

The common method of linear regression is the least squares method. It aims to find a polynomial, which best represents the function to interpret. So it tries to find the polynomial, which has the minimum difference with the function.


![maths](least_squares.png)


### Gradient descent method : Introduction to Machine Learning


The gradient descent method for linear regression is not useful. Yet it is the best example to introduce the gradient descent application in machine learning algorithms. In those algorithms, the variables are dynamically changing according to the error they induce. For a better understanding I invite you to alternate between the theory and the practice case as the theory may be a little uncomfortable. 

In the linear regression case we have two dynamic variables : `a` and `b`, which compose the polynomial `P = a * x + b` approaching the function to approximate.

Firstly `a` and `b` have a starting value, random or choosen. According to the cost function (which can be the euclidean function), the gradient for each variables are calculate.
Then the algorihtm changes step by step the variable's values to minimize the cost function. So each variables is changing according to its gradient. The changes are made until convergence. Here a mathematical explanation easier to understand :



It is important not to choose a learning rate to high or to low. 


![maths](gradient_descent.png)

On one hand, if you set a learning rate too low, learning will take too long.

On the other hand, if you set a learning rate too high, the variable's value jumps randomly whitout reaching the bottom of the cost function.

The aim is therefore to choose (experimentally most of the time) a learning rate that is neither too high nor too low

![learning rate](learning_rate.png)

## Thales stock prices case

The linear regression case here will be made through the Thales stock prices case since march of 2013. In this application we can use the linear regression to predict and explain the stock prices evolution. Explain means report the stock prices a posteriori.
You can find the data I use for the case in [this website](https://www.abcbourse.com/download/download.aspx?s=HOp).
 
The objective is to show by a linear regression the month-by-month global evolution of the Thales stock prices without noises. In this way we can also predict how the value will evolve. We will not make any prediction in this example, although we only need to "extend" the regression line.  

To be precise, it is not the stock prices at each 1st of the month but it is the prices each 40-45 days. It is not accurate to tell we will work with prices at each 1st of the month but we will consider it. It will not the the algorithms principles. Here a render of the .csv file we have : 

![render of the csv file](csv_render.png)

## Pseudo Code

### Least squares method

Least squares method for a 2D problem :

```
Calculate mean(X) and mean(Y) (X and Y are respectively months and stock prices in our exemaple)

Calculate COV(X, Y)

Calculate VAR(X)

Calculate a as a = COV(X, Y) / VAR(X)

Calculate b as b = mean(Y) - a * mean(X)

Return the polynomial a*x + b, which is the linear regerssion of the function f(X) = Y
```

### Gradient descent method

Gradient descent method for linear regression: (Be aware that cost = error²)

```
Give to a and b a starting value

Choose the learning rate L

While the minimum chosen cost or the number of maximum epoch chosen are not reaching
    
    Calculate the error (sum of each point's error)
    
    Calculate grad(a) and grad(b)
    
    Update a as a = a - grad(a) * L
    
    Update b as b = b - grad(b) * L
```

## Let's start with python

### Import

Firstly we need to import csv to extract the data from a .csv file. And we need to import matplotib for the render and its animation. 

```python
import csv
import matplotlib.pyplot as plt
import matplotlib.animation as animation
```

### Variables

As usual I made a region to change the variables to an easier understanding. Variables that can be manually changed are those, which are in uppercases. They are classed as variables for : 
* Choosing the linear regression method (binary choice)
* Stocking the a and b value of our linear polynominal a * x + b
* Choosing the vairables if gradient descent method is chosen. (`METHOD = 1`)
* Rendering the algorithm output

```python
dataset = {}

# choosing the linear regression method
    # 0 : least squares
    # 1 : gradient descent
METHOD = 1

# regression line coefficients
a, b  = 0, 0  # coefficients
STARTING_A = 0  # starting value for a in gradient descent method
STARTING_B = 0  # starting value for b in gradient descent method

# linear regression with gradient descent
EPOCHS = 10000  # number of epoch before stop running
LEARNING_RATE = 0.05  # learning rate
DISPLAY_EPOCH = 150  # display plot each DSIPLAY_EPOCH
epoch = 0  # to count epoch during the run


# plot
fig = plt.figure(1)  # to display the linear regression
ax = fig.add_subplot(1, 1, 1)  # to plot the linear regression
ani = None  # to animate the plot
started = False  # turn to True when Enter key is pressed
```

### Load dataset

Before to start our algorithms, we have to load a dataset. As I said before, the dataset here is the stock prices of Thales and the column we are interested in is the opening prices.

```python
def loadDataset(filename):
    """
    load a dataset from a filename
    :param filename: filename of the dataset
    """
    rows = []

    with open(filename, 'r') as f:  # reading csv file
        # creating a csv reader object
        csvreader = csv.reader(f)
        for row in csvreader:
            rows.append(row)
        rows.pop(0)  # remove the fields

    # keep months and 1st of month opening prices as a dict of arrays
    dataset['month'] = range(len(rows)) # begin the months to 0
    dataset['price'] = [float(row[2]) for row in rows] # extract opening prices
    return dataset
```

### Least squares method

The least squares method is easy to implement and is really near to the pseudo code written above. The function `least_squares()` calculate the `a` and `b` value of the polynomial `P(x) = a * x + b`, which approaches the best the stock prices. Then it returns the two values.

```python
def least_squares():
    """
    Return the coefficient of the linear regression using the least squares method
    :return: a, b (coefficients as f(x) = a*x + b)
    """
    length = len(dataset['month'])

    mean_month = 0
    mean_price = 0
    # calculate the means of the features
    for i in range(length):
        mean_price += dataset['price'][i]
        mean_month += dataset['month'][i]
    mean_price /= length
    mean_month /= length

    # as a reminder :
    # f(x) = COV(X,Y)/VAR(X)*x + b
    # b = mean(Y) - a * mean(X)
    cov_xy = 0
    var_x = 0

    # Calculation of COV(X, Y) and of VAR(X)
    for i in range(length):
        cov_xy += (dataset['month'][i] - mean_month) * (dataset['price'][i] - mean_price)
        var_x += (dataset['month'][i] - mean_month) ** 2

    # Calculation of the linear regression coefficients
    a = cov_xy / var_x
    b = mean_price - a * mean_month
    return a, b

```


### Gradient descent method

The `step_gradient_descent()` function returns the `a` and `b` coeffiecient updated one time. It will be the function to repeat to make the gradient descent working. The function takes four arguments : `X` and `Y` arrays, which respectively are in our cases the months array and the opening stock price of Thales associated to this date. It also take the `a` and `b` to update and to return. 

```python
def step_gradient_descent(X, Y, a, b):
    """
    Return the coefficient of the linear regression using the gradient descent method
    :return: a, b (coefficients as f(x) = a*x + b)
    :param X: X array
    :param Y: Y array
    :param a: coefficient a
    :param b: coefficient b
    :return: a, b, squared_error (coefficients as f(x) = a*x + b)
    """

    N = len(X)
    # calculate our current predictions
    predictions = [(a * X[i]) + b for i in range(N)]

    # calculate the errors
    error = [(Y[i] - predictions[i]) / N for i in range(N)]

    # calculate the gradients
    a_gradient = -(2 / N) * sum([X[i] * error[i] for i in range(N)])
    b_gradient = -(2 / N) * sum([error[i] for i in range(N)])

    # update the coefficients
    a -= LEARNING_RATE * a_gradient
    b -= LEARNING_RATE * b_gradient

    cost = sum([e**2 for e in error]) # if needed
    return a, b, cost
```

The `linear_regression_activate()` is launching the method chosen by the binary value `METHOD`. This function is a little special as it is called by the `matplotlib.animation` object. That's why there is the variable `frame_number` that I will not explain in this page. In my program, we will activate the linear regression method by pressing the Enter key. When the key is pressed, the `linear_regression_activate()` will run the chosen method of linear regression. 

As you may have seen, the `display()` is not written yet. I addition to that, the function, which links the Enter key to the start is also not written. I will be next. 

```python
def linear_regression_activate(frame_number):
    """
    Launch the chosen method of linear regression
    :param frame_number:
    """

    # wait for start
    if started:
        global epoch, a, b, squared_error
        if METHOD == 0:  # linear regression by least squares method
            a, b = least_squares()
            display()  # then display it
        elif METHOD == 1:  # linear regression by gradient descent method
            # initialize a and b coefficient
            a = STARTING_A
            b = STARTING_B

            # loop over the number of epochs
            while epoch < EPOCHS:
                # step by step gradient descent
                a, b, squared_error = step_gradient_descent(dataset['month'], dataset['price'], a, b)
                epoch += 1

                # then display the new update
                if epoch % DISPLAY_EPOCH == 0:
                    display()
```

### Display

That is the time to add the `display()` function to render the algorithm. It only shows the stock prices and waits for the Enter key too to display the linear regression.

```python
def display():
    """
    display the plot
    """
    # clear the plot
    ax.clear()

    # set the title and the axes
    ax.set_title('Thales stock market')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price (€)')

    # plot the stocks prices
    ax.plot(dataset['month'], dataset['price'], c='#2F9599')
    ax.set_ylim(0, max(dataset['price'])*1.1)  # not useful but more visual


    # waiting for start
    if started:
        # take the two extreme points of the line to plot it
        X = [min(dataset['month']), max(dataset['month'])]
        Y = [a * X[0] + b, a * X[1] + b]


        # set the label for the legend
        label = 'Linear Regression \n'
        label += 'Line : {0} * x + {1}\n'.format(round(a, 3), round(b, 3))
        if METHOD == 1:  # only for gradient descent method
            label += 'Epoch : {0}\n'.format(epoch)
            label += 'Learning Rate : {0}\n'.format(LEARNING_RATE)
            label += 'Squared Error : {0}'.format(round(squared_error,2))
        ax.plot(X, Y, c='#8800FF', label=label)

        # plot automatically the legend at the best position
        ax.legend(loc='best')

        # Uncomment the lines below to see the aim of gradient descent
        ''' 
        ls_a, ls_b = least_squares()
        X = [min(dataset['month']), max(dataset['month'])]
        Y = [ls_a * X[0] + ls_b, ls_a * X[1] + ls_b]
        ax.plot(X, Y, c='#FF77FF')
        '''
    # then plot everything
    fig.canvas.draw()

```

Finally, implement the `key_pressed()` function to activate the algorithm when Enter key is pressed.

```python
def key_pressed(event):
    """
    To start to run the programme by enter key
    :param event: key_press_event
    """
    if event.key == 'enter':
        global started
        started = not started

```
### Run it ! 

```python
if __name__ == '__main__':
    # load the dataset
    dataset = loadDataset("thales.csv")
    # connect to the key press event to start/pause the program
    fig.canvas.mpl_connect('key_press_event', key_pressed)
    # to animate the plot and launch the population update
    ani = animation.FuncAnimation(fig, linear_regression_activate)
    display()  # to visualize the function before linear regression
    plt.show()  # to activate the plot

```
