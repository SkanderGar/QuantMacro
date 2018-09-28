import matplotlib.pyplot as plt
import numpy as np
import os
os.chdir('C:/Users/DELL/Desktop/Quant_macro')
import useful_functions as uf

Tay1 = lambda x: uf.Taylor_pol(x, 1, 1, 0.321)
Tay2 = lambda x: uf.Taylor_pol(x, 1, 2, 0.321)
Tay5 = lambda x: uf.Taylor_pol(x, 1, 5, 0.321)
Tay20 = lambda x: uf.Taylor_pol(x, 1, 20, 0.321)
Truef = lambda x: x**321

x_eval = np.linspace(0, 4, 100)
x_eval = x_eval[1:-1]
 
Tay20_o = [Tay20(x_eval[i]) for i in range(len(x_eval))]
Tay5_o = [Tay5(x_eval[i]) for i in range(len(x_eval))]
Tay2_o = [Tay2(x_eval[i]) for i in range(len(x_eval))]
Tay1_o = [Tay1(x_eval[i]) for i in range(len(x_eval))]


f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
f.set_figheight(10)
f.set_figwidth(10)
ax1.plot(x_eval, Tay1_o, color = 'r', label = 'Taylor 1')
ax1.legend(loc='upper left')
ax1.set_xlabel('x')
ax1.set_ylabel('f(x)')
ax1.set_title('Expansion of Order 1')
ax2.plot(x_eval, Tay2_o, color = 'b', label = 'Taylor 2')
ax2.legend(loc='upper left')
ax2.set_xlabel('x')
ax2.set_ylabel('f(x)')
ax2.set_title('Expansion of Order 2')
ax3.plot(x_eval, Tay5_o, color = 'g', label = 'Taylor 5')
ax3.legend(loc='upper left')
ax3.set_xlabel('x')
ax3.set_ylabel('f(x)')
ax3.set_title('Expansion of Order 5')
ax4.plot(x_eval, Tay20_o, color = 'k', label = 'Taylor 20')
ax4.legend(loc='upper left')
ax4.set_xlabel('x')
ax4.set_ylabel('f(x)')
ax4.set_title('Expansion of Order 20')


realf = lambda x: x**(0.321) 
Rg = [realf(x_eval[i]) for i in range(len(x_eval))]
f1, ax5 = plt.subplots(1,1)
f1.set_figheight(5)
f1.set_figwidth(10)
ax5.set_yscale('log')
ax5.plot(x_eval, Tay1_o, color = 'r', label = 'Taylor 1')
ax5.plot(x_eval, Tay2_o, color = 'b', label = 'Taylor 2')
ax5.plot(x_eval, Tay5_o, color = 'g', label = 'Taylor 5')
ax5.plot(x_eval, Tay20_o, color = 'k', label = 'Taylor 20') 
ax5.plot(x_eval, Rg, color = 'gold', label = 'True function') 

ax5.legend(loc='upper left')
ax5.set_xlabel('x')
ax5.set_ylabel('f(x)')
ax5.set_title('The Taylor Expansions')



        