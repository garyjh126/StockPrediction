from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random


style.use('fivethirtyeight')

# xs = np.array([1,2,3,4,5,6], dtype=np.float64)
# ys = np.array([5,4,6,5,6,7], dtype=np.float64)

#hm - This is how many datapoints that we want in the set.
#variance - This will dictate how much each point can vary from the previous point. The more variance, the less-tight the data will be.
# step - how far on average to step up the y value on point.
# correlation positive, negative or none(Determined by step).
def create_dataset(hm, variance, step=2, correlation=False):
    val = 1
    ys = []
    for i in range(hm):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val+=step #default to 2
        elif  correlation and correlation == 'neg':
            val-=step
    xs = [i for i in range(len(ys))]

    return np.array(xs,dtype=np.float64), np.array(ys,dtype=np.float64)

def best_fit_slope_and_intercept(xs,ys):
    #Calculation for the slope of the best fit line
    m = ( ( (mean(xs)*mean(ys))- mean(xs*ys)) /
          ((mean(xs)*mean(xs)) - mean(xs*xs)))
    # Now we need to do a calculation for the y-intecept of the best fit line (b=mean(ys)-m*mean(xs))
    b = mean(ys) - m * mean(xs)
    return m, b

#Programming R Squared (Coefficient of determination). It measures how good of a fit is out best fit line.
#A big part of the calculation is the squared error.
def squared_error(ys_orig, ys_line):
    return sum((ys_line-ys_orig)**2)

def coefficient_of_determination(ys_orig,ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_regr/squared_error_y_mean)

#our assumption is that our r-squared/coefficient of determination should
# improve if we made the dataset a more tightly correlated dataset.
#  How would we do that?
# Simple: lower variance!
# Less variance should result in higher r-squared/coefficient of
# determination, higher variance = lower r squared.
xs, ys = create_dataset(40,40,2,correlation = 'pos')

m, b= best_fit_slope_and_intercept(xs,ys)
# Now we need to do a calculation for the y-intecept of the best fit line (b=mean(ys)-m*mean(xs))

regression_line = [(m*x)+b for x in xs]

predict_x = 8
predict_y = (m*predict_x)+b

r_squared = coefficient_of_determination(ys, regression_line)
print(r_squared)

plt.scatter(xs,ys)
plt.scatter(predict_x, predict_y, s=100,color = 'g')
plt.plot(xs,regression_line)
plt.show()


