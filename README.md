# Linear_Regression
```latex
\text{for } f \text{ the function that we are trying to approximate and } (x_i)_{i=0,\dots,n} \text{ for all its points. We are looking for a polynomial }\\ P_m(x) = a_0 + a_1x + \dots + a_mx \text{, which best approaches the function } f \text{. In other words we are looking for : } \\\\

\min_{a_0 \dots a_m}\left (\sum_{i=0}^{n} f(x_i) - P_m(x_i)\right) \\\\\\

\text{As in our case we are in a linear case we have : }\\\\

\min_{a, b \in\mathbb{R}}\left (\sum_{i=0}^{n} f(x_i) - (a(x_i) + b)\right) \\\\\\


\text{In practice, all we have to do is calculate the values as : } \\\\

a = {{cov(x_i,f(x_i))}\over{var(x_i)}} \text{     and    } b = \overline{f(x_i)} - m\overline{x_i}
```
![maths](https://latex.codecogs.com/gif.latex?\text{for&space;}&space;f&space;\text{&space;the&space;function&space;that&space;we&space;are&space;trying&space;to&space;approximate&space;and&space;}&space;(x_i)_{i=0,\dots,n}&space;\text{&space;for&space;all&space;its&space;points.&space;We&space;are&space;looking&space;for&space;a&space;polynomial&space;}\\&space;P_m(x)&space;=&space;a_0&space;&plus;&space;a_1x&space;&plus;&space;\dots&space;&plus;&space;a_mx&space;\text{,&space;which&space;best&space;approaches&space;the&space;function&space;}&space;f&space;\text{.&space;In&space;other&space;words&space;we&space;are&space;looking&space;for&space;:&space;}&space;\\\\&space;\min_{a_0&space;\dots&space;a_m}\left&space;(\sum_{i=0}^{n}&space;f(x_i)&space;-&space;P_m(x_i)\right)&space;\\\\\\&space;\text{As&space;in&space;our&space;case&space;we&space;are&space;in&space;a&space;linear&space;case&space;we&space;have&space;:&space;}\\\\&space;\min_{a,&space;b&space;\in\mathbb{R}}\left&space;(\sum_{i=0}^{n}&space;f(x_i)&space;-&space;(a(x_i)&space;&plus;&space;b)\right)&space;\\\\\\&space;\text{In&space;practice,&space;all&space;we&space;have&space;to&space;do&space;is&space;calculate&space;the&space;values&space;as&space;:&space;}&space;\\\\&space;a&space;=&space;{{cov(x_i,f(x_i))}\over{var(x_i)}}&space;\text{&space;and&space;}&space;b&space;=&space;\overline{f(x_i)}&space;-&space;m\overline{x_i})
