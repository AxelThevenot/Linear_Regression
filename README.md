# Linear_Regression
```latex
\text{for } f \text{ the function that we are trying to approximate and } (x_i)_{i=0,\dots,n} \text{ for all its points. We are looking for a polynomial }\\ P_m(x) = a_0 + a_1x + \dots + a_mx \text{, which best approaches the function } f \text{. In other words we are looking for : } \\\\

\min_{a_0 \dots a_m}\left (\sum_{i=0}^{n} f(x_i) - P_m(x_i)\right) \\\\\\

\text{As in our case we are in a linear case we have : }\\\\

\min_{a, b \in\mathbb{R}}\left (\sum_{i=0}^{n} f(x_i) - (a(x_i) + b)\right) \\\\\\


\text{In practice, all we have to do is calculate the values as : } \\\\

a = {{cov(x_i,f(x_i))}\over{var(x_i)}} \text{     and    } b = \overline{f(x_i)} - m\overline{x_i}
```
![maths](https://github.com/AxelThevenot/Linear_Regression/blob/master/CodeCogsEqn%20(4).gif)
