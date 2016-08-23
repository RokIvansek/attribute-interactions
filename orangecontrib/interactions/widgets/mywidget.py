from PyQt4 import QtCore, QtGui
from Orange.widgets.widget import OWWidget
from Orange.widgets import gui


class MyWidget(OWWidget):
    # Widget needs a name, or it is considered an abstract widget
    # and not shown in the menu.
    name = "Attribute interactions"
    icon = "icons/mywidget.svg"
    want_main_area = False

    def __init__(self):
        super().__init__()

        label = QtGui.QLabel("This is not Hello World anymore!")
        self.controlArea.layout().addWidget(
            label, QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
