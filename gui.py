import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import sys
from PyQt5.QtWidgets import QWidget, QMainWindow, QApplication, QAction, QDesktopWidget, QFileDialog, QPushButton, QToolTip,\
    QVBoxLayout, QHBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

import matplotlib.pyplot as plt

from bats import Batspy

from multiCH import get_all_ch, plot_multiCH_spectrogram

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
            self.statusBar().showMessage("%s selected... awaiting orders!" % ('.../' + '/'.join(self.fname.split('/')[-3:])))

    def load_single_Ch(self):
        loadSObj = QAction('&Load Single Channel', self)
        loadSObj.setShortcut('Ctrl+N')
        loadSObj.setStatusTip('Loads a single channel file')
        loadSObj.triggered.connect(self.click_singleCH)
        return loadSObj

    def click_singleCH(self):

        if not self.fname_selected:
            self.click_open()

        # close previous figure if plotted
        if self.multiCH_loaded or self.singleCH_loaded:
            self.figure.clear()

        bat = Batspy(self.fname, f_resolution=2 ** 9, overlap_frac=.70, dynamic_range=50, pcTape_rec=False)
        bat.compute_spectrogram()
        _, ax = bat.plot_spectrogram(ret_fig_and_ax=True, fig_input=self.figure, showit=False)

        # refresh canvas
        self.figure.tight_layout()
        self.canvas.draw()
        self.singleCH_loaded = True
        self.statusBar().showMessage("single channel: %s loaded" % ('.../' + '/'.join(self.fname.split('/')[-3:])))

        pass

    def click_multiCH(self):
        if not self.fname_selected:
            self.click_open()

        # close previous figure if plotted
        if self.multiCH_loaded or self.singleCH_loaded:
            self.figure.clear()

        all_fnames = get_all_ch(self.fname)

        # ToDo: Remake multiCH.py so that it is compatible with just plotting in the gui



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
        file.addAction(self.load_single_Ch())

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
        mwLength = 1800  # in Pixel

        # Establish main window size and title
        self.setGeometry(sWidth/2 - mwLength/2, sHeight/2 - mwLength/2, mwLength, mwLength)
        self.setWindowTitle('BATSpy')

        # Draw the main Canvas
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)

        # Create the Navigation Toolbar
        self.navToolbar = NavigationToolbar(self.canvas, self)

        # Select File button
        selFile = QPushButton('Select File (Ctrl+O)', self)
        selFile.released.connect(self.click_open)
        selFile.setToolTip('Select a File!')

        # Load singleCH button
        loadSCH = QPushButton('Load Single Channel (Ctrl+N)', self)
        loadSCH.released.connect(self.click_singleCH)
        # Set tool tip
        loadSCH.setToolTip('Analyze file as a <b>single channel<\b>')

        # Load multiCH button
        loadMCH = QPushButton('Load Multi Channel (Ctrl+M)', self)
        loadMCH.released.connect(self.click_multiCH)
        loadMCH.setToolTip('Analyze file as a <b>multi channel<\b>')

        # Set a horizontal box where the buttons will be placed
        hbox = QHBoxLayout()
        hbox.addWidget(selFile)
        hbox.addWidget(loadSCH)
        hbox.addWidget(loadMCH)

        layout = QVBoxLayout(self._main)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        layout.addWidget(self.navToolbar)
        layout.addLayout(hbox)

        self.show()


if __name__ == '__main__':

    spygui = QApplication(sys.argv)
    MainWindow = MainWindow()
    sys.exit(spygui.exec_())
