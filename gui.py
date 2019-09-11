import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QAction, QDesktopWidget, QFileDialog

from IPython import embed

class MainWindow(QMainWindow):
    
    def __init__(self, verbose=3):
        super().__init__()
        self.InitFunc()
        self.verbose = verbose

    def open(self):
        openObj = QAction('&Open', self)
        openObj.setShortcut('Ctrl+O')
        openObj.setStatusTip('Open a file browser dialog')
        openObj.triggered.connect(self.click_open)
        return openObj

    def click_open(self):
        fd = QFileDialog()
        fname = fd.getOpenFileName(self, 'Select File', '~/', 'Please select .wav files only (*.wav )')[0]
        if self.verbose == 3:
            print('opening file %s' %fname)
        self.statusBar().showMessage("%s selected" % fname)

    def quit(self):
        quitObj = QAction('&Quit', self)
        quitObj.setShortcut('Ctrl+Q')
        quitObj.setStatusTip('Quit BATSpy')
        quitObj.triggered.connect(self.close)
        return quitObj

    def InitFunc(self):

        # status bar
        self.statusBar().showMessage('Welcome to BATSpy!')
        
        # menu bar
        # File Submenu
        menubar = self.menuBar()
        file = menubar.addMenu("&File")
        file.addAction(self.open())
        file.addAction(self.quit())

        # View Submenu
        view = menubar.addMenu('&View')

        # Data Explorer Submenu
        dataex = menubar.addMenu('&Data Explorer')

        # Calls Submenu
        calls = menubar.addMenu('&Calls')
        
        # get current screen resolution
        sSize = QDesktopWidget().screenGeometry(-1)
        sWidth = sSize.width()
        sHeight = sSize.height()
        mwLength = 1500  # in Pixel

        # Establish main window size and title
        self.setGeometry(sWidth/2 - mwLength/2, sHeight/2 - mwLength/2, mwLength, mwLength)
        self.setWindowTitle('BATSpy')

        self.show()



if __name__ == '__main__':

    spygui = QApplication(sys.argv)
    MainWindow = MainWindow()
    sys.exit(spygui.exec_())
