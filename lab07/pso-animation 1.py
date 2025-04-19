import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
from pyswarms.utils.plotters.plotters import plot_contour
from pyswarms.utils.plotters.formatters import Mesher 


options = {'c1':0.2, 'c2':0.9, 'w':0.5} 
optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=2, options=options)
optimizer.optimize(fx.sphere, iters=50) 
# tworzenie animacji 
m = Mesher(func=fx.sphere) 
animation = plot_contour(pos_history=optimizer.pos_history, mesher=m, mark=(0, 0))
animation.save('plot0.gif', writer='imagemagick', fps=10)

#? Wysokie c1 i niskie c2 powodują, że cząstki poruszają się szybko w kierunku najlepszego rozwiązania, ale mogą nie eksplorować przestrzeni wystarczająco dobrze.
#? Niskie c1 i wysokie c2 powodują, że cząstki eksplorują przestrzeń bardziej, ale mogą nie koncentrować się na najlepszym rozwiązaniu.