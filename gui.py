import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QAction, QDesktopWidget, QFileDialog, QPushButton, QToolTip
from PyQt5.QtGui import QFont

from bats import Batspy

from IPython import embed

class MainWindow(QMainWindow):
    
    def __init__(self, verbose=3):
        super().__init__()
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
            self.statusBar().showMessage("%s selected" % self.fname)

    def load_singleCH(self):

        if not self.fname_selected:
            self.click_open()

        bat = Batspy(self.fname, f_resolution=2 ** 9, overlap_frac=.70, dynamic_range=50, pcTape_rec=False)
        bat.compute_spectrogram()
        bat.plot_spectrogram(showit=True)
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
        self.file_loaded = False

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
