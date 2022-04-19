from PyQt5 import QtCore, QtGui, QtWidgets
from Archivos.Principal import Principal

if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = Principal()
    window.show()
    app.exec_()
