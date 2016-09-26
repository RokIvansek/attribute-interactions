from orangecontrib.interactions.interactions import Interaction, Interactions
from Orange.widgets import gui, settings, widget, highcharts
from Orange.data import Table


class StackedBar(highcharts.Highchart):
    """
       StackedBar extends Highchart and just defines chart_type:
    """
    def __init__(self, **kwargs):
        super().__init__(chart_type='bar',
                         **kwargs)


class OWInteractionGraph(widget.OWWidget):
    """Example stacked bar plot visualization using Highcharts"""
    name = 'An interaction graph'
    description = 'An example stacked bar plot visualization using Highcharts.'
    icon = "icons/mywidget.svg"

    inputs = [("Interaction", Interaction, "set_interaction")]
    outputs = []

    graph_name = 'interaction'

    def __init__(self):
        super().__init__()
        self.interaction_data = None

        # Create an instance of StackedBar. Initial Highcharts configuration
        # can be passed as '_'-delimited keyword arguments. See Highcharts
        # class docstrings and Highcharts API documentation for more info and
        # usage examples.
        self.interaction_graph = StackedBar(title_text='Interaction graph example',
                                            plotOptions_series_stacking='normal',
                                            yAxis_min=0,
                                            yAxis_max=1,
                                            tooltip_shared=False)
        # Just render an empty chart so it shows a nice 'No data to display'
        # warning
        self.interaction_graph.chart()

        self.mainArea.layout().addWidget(self.interaction_graph)

    def set_interaction(self, inter_object):
        self.interaction_data = inter_object
        ab = self.interaction_data.rel_ig_ab
        if ab < 0:
            a = self.interaction_data.rel_ig_a
            b = self.interaction_data.rel_ig_b
            ab = -ab
        else:
            a = self.interaction_data.rel_ig_a - ab
            b = self.interaction_data.rel_ig_b - ab
            ab = self.interaction_data.rel_ig_ab
        options = dict(series=[])
        options['series'].append(dict(data=[a], name=self.interaction_data.a_name + ' info gain'))
        options['series'].append(dict(data=[ab], name='attribute interaction'))
        options['series'].append(dict(data=[b], name=self.interaction_data.b_name + ' info gain'))
        self.interaction_graph.chart(options)


def main():
    from PyQt4.QtGui import QApplication
    app = QApplication([])
    ow = OWInteractionGraph()

    d = Table("zoo")
    inter = Interactions(d)
    inter.interaction_matrix()
    int_object = inter.get_top_att(1)[0]

    ow.set_interaction(int_object)
    ow.show()
    app.exec_()

if __name__ == '__main__':
    main()

