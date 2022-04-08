import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import numpy as np

# build table for displaying data
index = ['Patients', 'Healthcare Professionals']
columns = ['μ', 'σ', 'Min.', 'Max.', 'Median', 'Q1', 'Q3']
data = [[3.881, 0.392, 2.72, 4.6, 3.98, 3.67, 4.14], [3.867, 0.450, 2.5, 4.6, 4, 3.6, 4.2]]
patient_df = pd.DataFrame(data, index, columns)
patient_df

# build normal curve showing patient score distributions
x_min = 2
x_max = 5.5

mean = 3.881
std = 0.392

x = np.linspace(x_min, x_max, 100)

y = scipy.stats.norm.pdf(x,mean,std)

plt.plot(x,y, color='blue')

plt.grid()

plt.xlim(x_min,x_max)
plt.ylim(0, 1.2)

plt.title('Patient Politeness Scores Distribution',fontsize=10)

plt.xlabel('')
plt.ylabel('')

#plt.axvline(x=2.72, linestyle='dashed', label='min.')
plt.vlines([2.72, 3.67, 3.98, 4.14, 4.6], 0, 1.09, linestyles='dashed', colors='red')
plt.text(2.65, 1.1, "min.")
plt.text(3.57, 1.1, "Q1")
plt.text(3.85, 1.1, "med.")
plt.text(4.1, 1.1, "Q3")
plt.text(4.5, 1.1, "max.")

plt.grid(False)
plt.savefig("normal_distribution.png")
plt.show()

# build normal curve showing clinician distributions
x_min = 2
x_max = 5.5

mean = 3.867
std = 0.450

x = np.linspace(x_min, x_max, 100)

y = scipy.stats.norm.pdf(x,mean,std)

plt.plot(x,y, color='blue')

plt.grid()

plt.xlim(x_min,x_max)
plt.ylim(0, 1.2)

plt.title('Healthcare Professional Politeness Scores Distribution',fontsize=10)

plt.xlabel('')
plt.ylabel('')

#plt.axvline(x=2.72, linestyle='dashed', label='min.')
plt.vlines([2.5, 3.6, 4, 4.2, 4.6], 0, 1.09, linestyles='dashed', colors='red')
plt.text(2.4, 1.1, "min.")
plt.text(3.5, 1.1, "Q1")
plt.text(3.85, 1.1, "med.")
plt.text(4.15, 1.1, "Q3")
plt.text(4.5, 1.1, "max.")

plt.grid(False)
plt.savefig("normal_distribution.png")
plt.show()
