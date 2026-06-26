import radage
import numpy as np
import matplotlib.pyplot as plt
t = np.linspace(0, 4500, 50)
r206_204, r207_204 = radage.sk_pb(t)
plt.figure(figsize=(6, 4))
plt.plot(r206_204, r207_204, linewidth=4)
plt.grid()
plt.xlabel(r'${}^{206}\mathrm{Pb}/{}^{204}\mathrm{Pb}$')
plt.ylabel(r'${}^{207}\mathrm{Pb}/{}^{204}\mathrm{Pb}$')
plt.title('Stacey and Kramers (1975) Common Lead Model')
plt.show()