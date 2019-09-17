import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import sys
from PyQt5.QtWidgets import QWidget, QMainWindow, QApplication, QAction, QDesktopWidget, QFileDialog, QPushButton, QToolTip,\
    QVBoxLayout
from PyQt5.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

import matplotlib.pyplot as plt

from bats import Batspy

from IPython import embed

class MainWindow(QMainWindow):
    
    def __init__(self, verbose=3):
        super().__init__()
        self._main = QWidget()
        self.setCentralWidget(self._main)

        self.verbose = verbose

        # Initialize function
        self.InitFunc()

    def open(self):
        openObj = QAction('&Open', self)
        openObj.setShortcut('Ctrl+O')
        openObj.setStatusTip('Open a file browser dialog')
        openObj.triggered.connect(self.click_open)
        return openObj

    def click_open(self):
        fd = QFileDialog()
        self.fname = fd.getOpenFileName(self, 'Select File', '~/', 'Please select .wav files only (*.wav )')[0]
        if self.verbose == 3:
            print('opening file %s' %self.fname)
        if len(self.fname) > 0:
            self.fname_selected = True
            self.statusBar().showMessage("%s selected" % ('.../' + '/'.join(self.fname.split('/')[-3:])))

    def load_singleCH(self):

        if not self.fname_selected:
            self.click_open()

        # close previous figure if plotted
        if self.multiCH_loaded or self.singleCH_loaded:
            self.fig.clear()

        bat = Batspy(self.fname, f_resolution=2 ** 9, overlap_frac=.70, dynamic_range=50, pcTape_rec=False)
        bat.compute_spectrogram()
        self.fig, ax = bat.plot_spectrogram(ret_fig_and_ax=True, showit=False)
        plt.draw()

        # plot
        self.plotWidget = FigureCanvas(self.fig)
        lay = QVBoxLayout(self._main)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.plotWidget)

        # pows, pks = bat.detect_calls(det_range=(80000, 150000), plot_in_spec=True)
        # embed()
        # plt.show()
        # quit()

    def quit(self):
        quitObj = QAction('&Quit', self)
        quitObj.setShortcut('Ctrl+Q')
        quitObj.setStatusTip('Quit BATSpy')
        quitObj.triggered.connect(self.close)
        return quitObj

    def InitFunc(self):

        # boolean flow control
        self.fname_selected = False
        self.singleCH_loaded = False
        self.multiCH_loaded = False
        self.fig = None

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

        # Draw the main Canvas

        # Load singleCH button
        loadSCh = QPushButton('Load Single Channel', self)
        loadSCh.move(mwLength - mwLength / 2, mwLength - 100)
        loadSCh.released.connect(self.load_singleCH)
        # Set tool tip
        QToolTip.setFont(QFont('Arial', 14))
        loadSCh.setToolTip('Analyze current file as a <b>single channel<\b>')

        self.show()



if __name__ == '__main__':

    spygui = QApplication(sys.argv)
    MainWindow = MainWindow()
    sys.exit(spygui.exec_())
