from orangecontrib.interactions.interactions import Interaction, Interactions
from orangecontrib.interactions.utils import load_mushrooms_data
from orangecontrib.bio.geo import GDS
from Orange.widgets import gui, settings, widget, highcharts
import Orange


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
            categories_left.append(o.var_a.name)
            categories_right.append(o.var_b.name)
            ab = o.rel_ig_ab
            if ab < 0:
                a = o.rel_ig_a
                b = o.rel_ig_b
                ab = -ab
                interaction_colors.append('#90ee7e') # #90ee7e #55BF3B
            else:
                a = o.rel_ig_a - ab
                b = o.rel_ig_b - ab
                interaction_colors.append('red') # #DF5353
            info_gains_left.append(b)
            interaction_gains.append(ab)
            info_gains_right.append(a)
        options = dict(series=[], xAxis=[])
        options['series'].append(dict(data=info_gains_left, name='Isolated attribute info gain', color='#7cb5ec'))
        options['series'].append(dict(data=interaction_gains,
                                      name='Interaction info gain',
                                      colorByPoint=True,
                                      colors = interaction_colors))
        options['series'].append(dict(data=info_gains_right, name='Isolated attribute info gain', color='#7cb5ec')) ##7798BF
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

    # d = Orange.data.Table('lenses')
    d = load_mushrooms_data()
    # gds = GDS("GDS1676")
    # d = gds.getdata()
    inter = Interactions(d)
    inter.interaction_matrix()
    int_object = inter.get_top_att(5)

    ow.set_interaction(int_object)
    ow.show()
    app.exec_()

if __name__ == '__main__':
    main()

