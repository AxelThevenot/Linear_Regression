# Linear_Regression
```latex
\text{for } f \text{ the function that we are trying to approximate and } (x_i)_{i=0,\dots,n} \text{ for all its points. We are looking for a polynomial }\\ P_m(x) = a_0 + a_1x + \dots + a_mx \text{, which best approaches the function } f \text{. In other words we are looking for : } \\\\

\min_{a_0 \dots a_m}\left (\sum_{i=0}^{n} f(x_i) - P_m(x_i)\right) \\\\\\

\text{As in our case we are in a linear case we have : }\\\\

\min_{a, b \in\mathbb{R}}\left (\sum_{i=0}^{n} f(x_i) - (a(x_i) + b)\right) \\\\\\


\text{In practice, all we have to do is calculate the values as : 

a = {{cov(x_i,f(x_i))}\over{var(x_i)}} \quad \text{and} \quad b = \overline{f(x_i)} - m\overline{x_i}
```
![maths](least_squares.gif)
```
 \\\text{For }J \text{ the cost of the function  we have : } \\\\
J(a_0,\dots,a_m)=\frac{1}{n}\sum_{i=1}^{n} \left(f(x_i)-P_m(x_i)\right)^2\\\\

\text{In the linear case we have : }\\\\
P_2(x_i) = a(x_i) + b\\\\


\text{So the gradients are : }\\\\
\displaystyle \frac{\partial}{\partial a}J(a, b) = \frac{-2}{n}\sum_{i=1}^{n} \left(f(x_i)- (ax_i + b)\right).x_i = \frac{-2}{n}\sum_{i=1}^{n} \left(f(x_i)-P_2(x_i)\right).x_i\quad\text{and}\\\\
\displaystyle \frac{\partial}{\partial b}J(a, b) = \frac{-2}{n}\sum_{i=1}^{n} \left(f(x_i)- (ax_i + b)\right) = \frac{-2}{n}\sum_{i=1}^{n} \left(f(x_i)-P_2(x_i)\right)\\

\\

\text{The gradient descent algorithme need a learning rate } L \text{ and repeat the folowing operations until convergence : }\\\\

\left\{a:=a- L\frac{\partial}{\partial a}J(a, b)\right\} \quad and \quad \left\{b:=b - L\frac{\partial}{\partial b}J(a, b)\right\}
```
![maths](gradient_descent.gif)
