import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import sys
from PyQt5.QtWidgets import QWidget, QMainWindow, QApplication, QAction, QDesktopWidget, QFileDialog, QPushButton, QToolTip,\
    QVBoxLayout, QHBoxLayout, QGridLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

import matplotlib.pyplot as plt

from bats import Batspy

from multiCH import get_all_ch, load_all_channels, plot_multiCH_spectrogram, plot_calls_in_spectrogram

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
            self.statusBar().showMessage("%s selected... NOW LOAD EITHER SINGLE OR MULTI CHANNEL!" % ('.../' + '/'.join(self.fname.split('/')[-3:])))

    def load_single_Ch(self):
        loadSObj = QAction('Load &Single Channel', self)
        loadSObj.setShortcut('Ctrl+N')
        loadSObj.setStatusTip('Loads a single channel file')
        loadSObj.triggered.connect(self.click_singleCH)
        return loadSObj

    def click_singleCH(self):

        if not self.fname_selected:
            self.click_open()

        self.statusBar().showMessage('Loading Single Channel...')

        # close previous figure
        self.figure.clear()

        bat = Batspy(self.fname, f_resolution=2 ** 9, overlap_frac=.70, dynamic_range=50, pcTape_rec=False)
        bat.compute_spectrogram()
        _, ax = bat.plot_spectrogram(ret_fig_and_ax=True, fig_input=self.figure, showit=False)

        # refresh canvas
        self.figure.tight_layout()
        self.canvas.draw()
        self.singleCH_loaded = True
        self.multiCH_loaded = False
        self.statusBar().showMessage("single channel: %s loaded" % ('.../' + '/'.join(self.fname.split('/')[-3:])))

        pass

    def load_multiCH(self):
        loadMObj = QAction('Load &Multi Channel', self)
        loadMObj.setShortcut('Ctrl+M')
        loadMObj.setStatusTip('Loads all channels of the same recording simultaneously')
        loadMObj.triggered.connect(self.click_multiCH)
        return loadMObj

    def click_multiCH(self):

        # ToDo: Perform this method as a separate Thread?
        if not self.fname_selected:
            self.click_open()

        self.statusBar().showMessage('Loading Multi Channel...')

        # close previous figure
        self.figure.clear()

        specs, spec_time, spec_freq = load_all_channels(self.fname)
        plot_multiCH_spectrogram(specs, spec_time, spec_freq, self.fname, input_fig=self.figure)

        # refresh canvas
        self.figure.tight_layout()
        self.canvas.draw()
        self.multiCH_loaded = True
        self.singleCH_loaded = False
        self.statusBar().showMessage("Multi channel: %s loaded" % ('.../' + '/'.join(self.fname.split('/')[-3:])))

        # ToDo: numpy.memmap for loading a huge file directly from the hard drive!!! Do this for the calculated specs

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
        file.addAction(self.load_multiCH())

        # View Submenu
        view = menubar.addMenu('&View')

        # Data Explorer Submenu
        dataex = menubar.addMenu('&Data Explorer')

        # Calls Submenu
        calls = menubar.addMenu('&Calls')

        # ToDo: integrate the call detection. check if multiCH or singleCH is loaded first
        # calls.addAction(self.detect_calls())

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
        # ToDo: create a Handmade Navigation Toolbar with shortcuts
        # self.navToolbar = NavigationToolbar(self.canvas, self)

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
        loadMCH.setToolTip('Search for matching recordings and show all files simultaneously. <b>Multi channel<\b>')

        # Set a layout grid where the plot and buttons will be placed
        self.central_widget = QWidget(self)
        layGrid = QGridLayout()
        layGrid.addWidget(self.canvas, 0, 0, 4, 5)
        layGrid.addWidget(selFile, 4, 0, 1, 1)
        layGrid.addWidget(loadSCH, 4, 1, 1, 1)
        layGrid.addWidget(loadMCH, 4, 2, 1, 1)

        self.central_widget.setLayout(layGrid)
        self.setCentralWidget(self.central_widget)

        self.show()


if __name__ == '__main__':

    spygui = QApplication(sys.argv)
    MainWindow = MainWindow()
    sys.exit(spygui.exec_())
