from orangecontrib.interactions.interactions import Interaction, Interactions
from Orange.widgets import gui, settings, widget, highcharts
import Orange
import numpy as np

def load_mushrooms_data():
    # shrooms_data = np.array(np.genfromtxt("../datasets/agaricus-lepiota.data", delimiter=",", dtype=str))
    shrooms_data = np.array(np.genfromtxt("https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data", delimiter=",", dtype=str))
    # Convert mushroom data from strings to integers
    names = ['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment', 'gill-spacing', 'gill-size',
            'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring',
             'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 'ring-type',
             'spore-print-color', 'population', 'habitat']
    for i in range(len(shrooms_data[0, :])):
        u, ints = np.unique(shrooms_data[:, i], return_inverse=True)
        shrooms_data[:, i] = ints
    shrooms_data = shrooms_data.astype(np.float32)
    Y_shrooms = shrooms_data[:, 0]
    X_shrooms = shrooms_data[:, 1:]
    domain = Orange.data.Domain([Orange.data.DiscreteVariable(names[i-1]) for i in range(1,X_shrooms.shape[1]+1)],
                                Orange.data.DiscreteVariable("edible"))
    data = Orange.data.Table(domain, X_shrooms, Y_shrooms)  # Make an Orange.Table object
    return data

class StackedBar(highcharts.Highchart):
    """
       StackedBar extends Highchart and just defines chart_type:
    """
    def __init__(self, **kwargs):
        super().__init__(chart_type='bar',
                         **kwargs)


class OWInteractionGraph(widget.OWWidget):
    """Interactions visualization using Highcharts"""
    name = 'An interaction graph'
    description = 'An example stacked bar plot visualization using Highcharts.'
    icon = "icons/mywidget.svg"

    inputs = [("Interaction", Interaction, "set_interaction")]
    outputs = []

    graph_name = 'interaction'

    def __init__(self):
        super().__init__()
        self.data = None

        # Create an instance of StackedBar. Initial Highcharts configuration
        # can be passed as '_'-delimited keyword arguments. See Highcharts
        # class docstrings and Highcharts API documentation for more info and
        # usage examples.
        self.interaction_graph = StackedBar(title_text='Interactions graph',
                                            plotOptions_series_stacking='normal',
                                            yAxis_min=0,
                                            yAxis_max=1,
                                            yAxis_title_text='Relative information gain',
                                            tooltip_shared=False)
        # Just render an empty chart so it shows a nice 'No data to display'
        # warning
        self.interaction_graph.chart()

        self.mainArea.layout().addWidget(self.interaction_graph)

    def set_interaction(self, interactions_list):
        self.data = interactions_list
        categories_left = []
        categories_right = []
        info_gains_left = []
        info_gains_right = []
        interaction_gains = []
        interaction_colors = []
        for o in self.data:
            categories_left.append(o.a_name)
            categories_right.append(o.b_name)
            ab = o.rel_ig_ab
            if ab < 0:
                a = o.rel_ig_a
                b = o.rel_ig_b
                ab = -ab
                interaction_colors.append('green')
            else:
                a = o.rel_ig_a - ab
                b = o.rel_ig_b - ab
                interaction_colors.append('red')
            info_gains_left.append(b)
            interaction_gains.append(ab)
            info_gains_right.append(a)
        options = dict(series=[], xAxis=[])
        options['series'].append(dict(data=info_gains_left, name='Isolated attribute info gain', color='blue'))
        options['series'].append(dict(data=interaction_gains,
                                      name='Interaction info gain',
                                      colorByPoint=True,
                                      colors = interaction_colors))
        options['series'].append(dict(data=info_gains_right, name='Isolated attribute info gain', color='blue'))
        options['xAxis'].append(dict(categories=categories_left,
                                     labels = dict(step=1)))
        options['xAxis'].append(dict(categories=categories_right,
                                     opposite=True,
                                     linkedTo=0,
                                     labels = dict(step=1)))
        self.interaction_graph.chart(options)


def main():
    from PyQt4.QtGui import QApplication
    app = QApplication([])
    ow = OWInteractionGraph()

    # d = Orange.data.Table('zoo')
    d = load_mushrooms_data()
    inter = Interactions(d)
    inter.interaction_matrix()
    int_object = inter.get_top_att(5)

    ow.set_interaction(int_object)
    ow.show()
    app.exec_()

if __name__ == '__main__':
    main()

