#a10Script.py
#hacked together by Abe
import numpy as np
import a10 as lf
import imageIO as io
io.baseInputPath = './'

##This assumes that you unzip the chessNumpy.zip into the same directory
# chess3x3 = np.load('./Input/chess/chess3x3.npy')
chess5x5 = np.load('./Input/chess/chess5x5.npy')
# chess17x17 = np.load('./Input/chess/chess17x17.npy')

#
# bugStack = np.load('./Input/BugStack.npy')
# lego = np.load('./Input/lego17x17.npy')

# lytro1 = np.load('counter1.npy')

def testApertureView(LF, outname):
    io.imwrite(lf.apertureView(LF), outname+'.png')

def testEpiSlice(LF, y, outname):
    io.imwrite(lf.epiSlice(LF, y), outname+'.png')

def testRefocusLF(LF, focus, outname):
    aperture = LF.shape[0]
    io.imwrite(lf.refocusLF(LF, focus, aperture), outname+'.png')

def testRackFocus(LF, outname, nIms = 15, minPar=-7.0, maxPar=2.0):
    aperture = LF.shape[0]
    fstack = lf.rackFocus(LF, aperture, nIms, minPar, maxPar)
    fullname = outname+'_min'+str(minPar)+'_max'+str(maxPar)+'nIms_'+str(nIms)
    printFocalStack(fstack, fullname+'_')
    return fstack, fullname

def printFocalStack(FS, outname):
    for i in xrange(FS.shape[0]):
        io.imwrite(FS[i], outname+str(i)+'.png')

def saveNP(A, outname):
    np.save('./myLFs/'+outname+'.npy', A)

def printSharpnessStack(FS, outname):
    printFocalStack(100*lf.sharpnessStack(FS), outname)

def testFullFocusLinear(FS, outname, exponent=3.0, sigma=1.0):
    allfocus, depthmap = lf.fullFocusLinear(FS, exponent, sigma)
    io.imwrite(allfocus, outname+'AllFocus.png')
    io.imwrite(depthmap, outname+'DepthMap.png')

def testSharpness():
    for i in xrange(chess5x5.shape[0]):
        sharp = 100*lf.sharpnessMap(chess5x5[0][i])
        print 'sharp', sharp.shape
        io.imwrite(sharp, 'sharp'+str(i)+'.png')

# testEpiSlice(lego, 100, "legoEpiSlice")
# legoStack, legoStackName = testRackFocus(lego, "legoStack", 15, -40.0, 8.0)
# testFullFocusLinear(legoStack, "legoFullFocus")

#testApertureView(chess3x3, "chess3x3ApertureViews")
testApertureView(chess5x5, "chess5x5ApertureViews")
#testApertureView(chess17x17, "chess17x17ApertureViews")
testEpiSlice(chess5x5, 100, "chess5x5Epislicey100")

#testSharpness()
chessStack5, chessStack5Name = testRackFocus(chess5x5, "chess5x5_new")
# saveNP(chessStack5, chessStack5Name)
# chessStack17, chessStack17Name = testRackFocus(chess17x17, "chess17x17_new")
# saveNP(chessStack17, chessStack17Name)

#chessStack17 = np.load('myLFs/chess17x17_min-7.0_max2.0nIms_15.npy')
# printFocalStack(chessStack5, "testChessStack")
printSharpnessStack(chessStack5, 'testChessSharpness')
testFullFocusLinear(chessStack5, "chessStack5")
# testFullFocusLinear(bugStack, "bugStack")
#testFullFocusLinear(lytro1, "lytro1")
