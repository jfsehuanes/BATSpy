import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import sys
import numpy as np
from PyQt5.QtWidgets import QWidget, QMainWindow, QApplication, QAction, QDesktopWidget, QFileDialog, QPushButton, QToolTip,\
    QVBoxLayout, QHBoxLayout, QGridLayout, qApp
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt, QEvent
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

import matplotlib.pyplot as plt

from bats import Batspy

from multiCH import get_all_ch, load_all_channels, plot_multiCH_spectrogram, plot_calls_in_spectrogram

from IPython import embed


class PlotClass:
    
    def __init__(self):

        # Draw the main Canvas
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.figure.clear()  # Clear the figure so that the initial screen is a blank screen

        self.ax = None

        self.xlim = None
        self.ylim = None

        self.init_xlim = None
        self.init_ylim = None

        # Set important variables to None
        self.fname = None

    def zoom_out(self):
        xlim = self.xlim
        ylim = self.ylim

        new_xlim = (xlim[0] - np.diff(xlim)[0] * 0.25, xlim[1] + np.diff(xlim)[0] * 0.25)
        new_ylim = (ylim[0] - np.diff(ylim)[0] * 0.02, ylim[1] + np.diff(ylim)[0] * 0.02)
        self.ylim = new_ylim
        self.xlim = new_xlim

        self.ax.set_xlim(*new_xlim)
        self.ax.set_ylim(*new_ylim)

        self.figure.canvas.draw()

    def zoom_in(self):
        xlim = self.xlim
        ylim = self.ylim

        new_xlim = (xlim[0] + np.diff(xlim)[0] * 0.25, xlim[1] - np.diff(xlim)[0] * 0.25)
        new_ylim = (ylim[0] + np.diff(ylim)[0] * 0.02, ylim[1] - np.diff(ylim)[0] * 0.02)
        self.ylim = new_ylim
        self.xlim = new_xlim

        self.ax.set_xlim(*new_xlim)
        self.ax.set_ylim(*new_ylim)

        self.figure.canvas.draw()

    def zoom_home(self):
        new_xlim = self.init_xlim
        new_ylim = self.init_ylim
        self.ylim = new_ylim
        self.xlim = new_xlim

        self.ax.set_xlim(*new_xlim)
        self.ax.set_ylim(*new_ylim)

        self.figure.canvas.draw()

    def move_right(self):
        xlim = self.xlim

        new_xlim = (xlim[0] + np.diff(xlim)[0] * 0.25, xlim[1] + np.diff(xlim)[0] * 0.25)
        self.xlim = new_xlim

        self.ax.set_xlim(*new_xlim)

        self.figure.canvas.draw()

    def move_left(self):
        xlim = self.xlim

        new_xlim = (xlim[0] - np.diff(xlim)[0] * 0.25, xlim[1] - np.diff(xlim)[0] * 0.25)
        self.xlim = new_xlim

        self.ax.set_xlim(*new_xlim)

        self.figure.canvas.draw()

    def move_up(self):
        ylim = self.ylim

        new_ylim = (ylim[0] + np.diff(ylim)[0] * 0.25, ylim[1] + np.diff(ylim)[0] * 0.25)
        self.ylim = new_ylim

        self.ax.set_ylim(*new_ylim)
        self.figure.canvas.draw()

    def move_down(self):
        ylim = self.ylim

        new_ylim = (ylim[0] - np.diff(ylim)[0] * 0.25, ylim[1] - np.diff(ylim)[0] * 0.25)
        self.ylim = new_ylim

        self.ax.set_ylim(*new_ylim)
        self.figure.canvas.draw()

    def plot_singleCH(self):

        # close previous figure
        self.figure.clear()

        bat = Batspy(self.fname, f_resolution=2 ** 9, overlap_frac=.70, dynamic_range=50, pcTape_rec=False)
        bat.compute_spectrogram()

        _, ax = bat.plot_spectrogram(ret_fig_and_ax=True, input_fig=self.figure, interpolation_type='hanning',
                                     showit=False)
        self.ax = ax[0]

        if self.xlim is None:
            self.init_xlim = ax[0].get_xlim()
            self.init_ylim = ax[0].get_ylim()
            self.xlim = ax[0].get_xlim()
            self.ylim = ax[0].get_ylim()
        else:
            self.xlim = ax[0].get_xlim()
            self.ylim = ax[0].get_ylim()

        # refresh canvas
        self.figure.tight_layout()
        self.canvas.draw()

        pass

    def plot_multiCH(self):

        # close previous figure
        self.figure.clear()

        specs, spec_params = load_all_channels(self.fname)
        _, multi_spec_ax, all_calls_ax = plot_multiCH_spectrogram(specs, spec_params, self.fname,
                                                               interpolation_type='hanning',
                                                               input_fig=self.figure,
                                                               ret_fig_chAxs_and_callAx=True)
        self.ax = multi_spec_ax[0]
        if self.xlim is None:
            self.init_xlim = multi_spec_ax[0].get_xlim()
            self.init_ylim = multi_spec_ax[0].get_ylim()
            self.xlim = multi_spec_ax[0].get_xlim()
            self.ylim = multi_spec_ax[0].get_ylim()
        else:
            self.xlim = multi_spec_ax[0].get_xlim()
            self.ylim = multi_spec_ax[0].get_ylim()

        # ToDo: Do here the save and afterwards load stuff
        # np.save('temp_files/test_multi.npy', specs)
        # h = np.load('temp_files/test_multi.npy')

        # embed()
        # quit()

        # refresh canvas
        self.figure.tight_layout()
        self.canvas.draw()
        pass


class MainWindow(QMainWindow):
    
    def __init__(self, parent=None, verbose=3):
        super(MainWindow, self).__init__(parent)
        self._main = QWidget()
        self.setCentralWidget(self._main)

        self.verbose = verbose

        # Insert the PlotClass
        self.Plot = PlotClass()

        # Insert the toolbar init
        self.initActions()
        self.init_ZoomToolBar()

        # # Install the EventFilter
        # qApp.installEventFilter(self)
        
        # Initialize function
        self.InitFunc()

    def init_ZoomToolBar(self):
        zToolBar = self.addToolBar('zTB')

        # zToolBar.addAction(self.Act_interactive_zoom)
        zToolBar.addAction(self.Act_zoom_out)
        zToolBar.addAction(self.Act_zoom_in)
        zToolBar.addAction(self.Act_zoom_home)
        # zToolBar.addAction(self.Act_arrowkeys)

    def initActions(self):

        self.Act_zoom_out = QAction(QIcon('symbols/zoomout.png'), 'Zoom -', self)
        self.Act_zoom_out.triggered.connect(self.Plot.zoom_out)
        self.Act_zoom_out.setShortcut('Ctrl+-')
        self.Act_zoom_out.setEnabled(False)

        self.Act_zoom_in = QAction(QIcon('symbols/zoomin.png'), 'Zoom +', self)
        self.Act_zoom_in.triggered.connect(self.Plot.zoom_in)
        self.Act_zoom_in.setShortcut('Ctrl++')
        self.Act_zoom_in.setEnabled(False)

        self.Act_zoom_home = QAction(QIcon('symbols/zoom_home.png'), 'Zoom Home', self)
        self.Act_zoom_home.triggered.connect(self.Plot.zoom_home)
        self.Act_zoom_home.setShortcut('Ctrl+h')
        self.Act_zoom_home.setEnabled(False)

        self.Act_interactive_zoom = QAction(QIcon('symbols/zoom.png'), 'Zoom Select', self)
        self.Act_interactive_zoom.setCheckable(True)
        self.Act_interactive_zoom.setEnabled(False)

        # self.Act_arrowkeys = QAction(QIcon('symbols/arrowkeys.png'), 'Activate arrorw keys', self)
        # self.Act_arrowkeys.setCheckable(True)
        # self.Act_arrowkeys.setEnabled(False)

    def enable_plot_buttons(self):
        self.Act_zoom_out.setEnabled(True)
        self.Act_zoom_in.setEnabled(True)
        self.Act_zoom_home.setEnabled(True)
    #     self.Act_arrowkeys.setEnabled(True)
    #
    # # Create an eventfilter for panning with the ArrowKeys. Need EventFilter to avoid conflict with arrow-keys
    # def eventFilter(self, source, event):
    #     if event.type() == QEvent.KeyPress:
    #         if self.Act_arrowkeys.isChecked():
    #             if event.key() == Qt.Key_Right:
    #                 self.Plot.move_right()
    #                 return True
    #             elif event.key() == Qt.Key_Left:
    #                 self.Plot.move_left()
    #                 return True
    #             elif event.key() == Qt.Key_Up:
    #                 self.Plot.move_up()
    #                 return True
    #             elif event.key() == Qt.Key_Down:
    #                 self.Plot.move_down()
    #                 return True
    #
    #     return super(MainWindow, self).eventFilter(source, event)

    def open(self):
        openObj = QAction('&Open', self)
        openObj.setShortcut('Ctrl+O')
        openObj.setStatusTip('Open a file browser dialog')
        openObj.triggered.connect(self.click_open)
        return openObj

    def click_open(self):
        fd = QFileDialog()
        self.fname = fd.getOpenFileName(self, 'Select File', '~/', 'Please select .wav files only (*.wav )')[0]
        self.Plot.fname = self.fname
        if self.verbose == 3:
            print('opening file %s' % self.fname)
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

        self.Plot.plot_singleCH()

        self.singleCH_loaded = True
        self.multiCH_loaded = False
        self.statusBar().showMessage("single channel: %s loaded" % ('.../' + '/'.join(self.fname.split('/')[-3:])))

        self.enable_plot_buttons()
        pass

    def load_multiCH(self):
        loadMObj = QAction('Load &Multi Channel', self)
        loadMObj.setShortcut('Ctrl+M')
        loadMObj.setStatusTip('Loads all channels of the same recording simultaneously')
        loadMObj.triggered.connect(self.click_multiCH)
        return loadMObj

    def click_multiCH(self):

        if not self.fname_selected:
            self.click_open()

        self.statusBar().showMessage('Loading Multi Channel...')

        self.Plot.plot_multiCH()

        self.multiCH_loaded = True
        self.singleCH_loaded = False
        self.statusBar().showMessage("Multi channel: %s loaded" % ('.../' + '/'.join(self.fname.split('/')[-3:])))

        self.enable_plot_buttons()
        # ToDo: numpy.memmap for loading a huge file directly from the hard drive!!! Do this for the calculated specs.
        # ToDo: First compute the spectrogram, then save it as a numpy file and finally read it with memmap

    def detect_calls(self):


        pass

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

        # Create the Navigation Toolbar
        # ToDo: create a Handmade Navigation Toolbar with shortcuts
        # self.navToolbar = NavigationToolbar(self.Plot.canvas, self)

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
        layGrid.addWidget(self.Plot.canvas, 0, 0, 4, 5)
        layGrid.addWidget(selFile, 4, 0, 1, 1)
        layGrid.addWidget(loadSCH, 4, 1, 1, 1)
        layGrid.addWidget(loadMCH, 4, 2, 1, 1)

        self.central_widget.setLayout(layGrid)
        self.setCentralWidget(self.central_widget)

        self.show()


if __name__ == '__main__':

    spygui = QApplication(sys.argv)
    MainWindow = MainWindow()
    # ToDo: remove tmp files when closing app
    sys.exit(spygui.exec_())
