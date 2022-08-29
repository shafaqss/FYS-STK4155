from imageio import imread
from numpy.random import normal, uniform
from realparta import*
from bootstrap import bootstrap
from kfold import kfold_cross_validation
# Load the terrain
terrain = imread('SRTM_data_Norway_1.tif')

# Show the terrain
def plot_terrain():
    plt.figure()
    plt.title('Terrain over Norway 1')
    plt.imshow(terrain1, cmap='gray')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig("data.png", format='png')
    #plt.show()

name = "terrain"; N = 100; m = 15 # polynomial order
degree = m;
terrain = terrain[:N,:N]

# Creates mesh of image pixels
x = np.linspace(0,1, np.shape(terrain)[0])
y = np.linspace(0,1, np.shape(terrain)[1])

x, y = np.meshgrid(x,y)
#X = create_design_matrix(x_mesh, y_mesh, m)
z = terrain.ravel()
B = 10; k = 10;

"""
imagename must have .png ending eg, lasso.png, the functions have arguments:

bootstrap(B, x, y, z, reg, lambda_, imagename, degree,name=None, plot=False)
kfold_cross_validation(k, x, y, z, degree, reg, lambda_, imagename, name=None, plot=False)
"""
#bootstrap(B, x, y, z, "ols", 0, "olsterrain.png", degree, name, plot=True)
#bootstrap(B, x, y, z, "ridge", 0.5, "olsterrain.png", degree, name, plot=True)
#bootstrap(B, x, y, z, "lasso", 0.5, "olsterrain.png", degree, name, plot=True)

#kfold_cross_validation(k, x, y, z, degree, "ols", 0,"lassofrank.png", name, plot=True)
#kfold_cross_validation(k, x, y, z, degree, "ridge", 0.9,"lassofrank.png", name, plot=True)
kfold_cross_validation(k, x, y, z, degree, "lasso", 4 ,"lassofrank.png", name, plot=True)
