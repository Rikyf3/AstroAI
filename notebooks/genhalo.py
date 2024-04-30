import math
import traceback
import numpy as np
import numbers
import cv2 as cv
import matplotlib.pyplot as plt
import lmfit
from lmfit.lineshapes import gaussian2d, lorentzian
from scipy.ndimage import gaussian_filter
from auto_stretch import stretch as AutoStretch
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder, IRAFStarFinder
from photutils.detection import find_peaks
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils.aperture import CircularAperture


class Image:
    def __init__(self, img, sigma = 3.0):
        if len(img.shape) !=2 and len(img.shape) != 3:
            raise ValueError("Unknown image dimensions")

        self.__origImg = img.copy()
        self.__stretchedOrigImg = None
        self.__origLum = None
        self.__stretchedOrigLum = None

        self.__img = None
        self.__stretchedImg = None
        self.__updateStretchedImg = True
        self.__lum = None
        self.__updateLum = True
        self.__stretchedLum = None
        self.__updateStretchedLum = True

        if len(img.shape) == 3:
            if img.shape[0] != 1 and img.shape[0] != 3:
                raise ValueError("Third image dimension must have length 1 or 3")

        self.__sigma = sigma
        self.__finderOptions = dict()

        self.__stats = None
        self.__updateStats = True
        self.__lumStats = None
        self.__updateLumStats = True
        self.__origStats = None
        self.__origLumStats = None

        self.__stars = None

    @property
    def nx(self):
        return self.img.shape[-2]
    @property
    def ny(self):
        return self.img.shape[-1]

    def __setUpdate(self):
        self.__updateLum = True
        self.__updateStats = True
        self.__updateLumStats = True
        self.__updateStretchedImg = True
        self.__updateStretchedLum = True

    @property
    def stretchedOrigImg(self):
        if self.__stretchedOrigImg is None:
            stretch = AutoStretch.Stretch()
            self.__stretchedOrigImg = stretch.stretch(self.origImg)
        return self.__stretchedOrigImg

    @property
    def stretchedOrigLum(self):
        if self.__stretchedOrigLum is None:
            stretch = AutoStretch.Stretch()
            self.__stretchedOrigLum = stretch.stretch(self.origLum)
        return self.__stretchedOrigLum

    @property
    def stretchedImg(self):
        if self.__updateStretchedImg or self.__stretchedImg is None:
            stretch = AutoStretch.Stretch()
            self.__stretchedImg = stretch.stretch(self.img)
            self.__updateStretchedImg = False
        return self.__stretchedImg

    @property
    def stretchedLum(self):
        if self.__updateStretchedLum or self.__stretchedLum is None:
            stretch = AutoStretch.Stretch()
            self.__stretchedLum = stretch.stretch(self.lum)
            self.__updateStretchedLum = False
        return self.__stretchedLum


    @property
    def origImg(self):
        return self.__origImg

    @property
    def origLum(self):
        if self.__origLum is None:
            if self.nChannels == 3:
                self.__origLum = 1./3. * (self.__origImg[0] + self.__origImg[1] + self.__origImg[2])
            else:
                self.__origLum = self.__origImg # Reference
        return self.__origLum

    @property
    def img(self):
        if self.__img is None:
            self.__img = self.__origImg.copy()
        return self.__img

    def __calcLum(self):
        if self.nChannels == 3:
            self.__lum = 1./3. * (self.img[0] + self.img[1] + self.img[2])
        else:
            self.__lum = self.img
        self.__updateLum = False

    @property
    def lum(self):
        if self.__updateLum or self.__lum is None:
            self.__calcLum()
        return self.__lum

    def __calcOrigLumStats(self):
        mean, median, std = sigma_clipped_stats(self.origLum, sigma=self.__sigma)
        self.__origLumStats = {
                'mean': mean,
                'median': median,
                'std': std
                }

    @property
    def origLumStats(self):
        if self.__origLumStats is None:
            self.__calcOrigStats()
        return self.__origLumStats

    def __calcOrigStats(self):
        if self.nChannels == 2:
            self.__origStats = self.origLumStats
            return
        self.__origStats = {
                'mean': np.empty(self.nChannels),
                'median': np.empty(self.nChannels),
                'std': np.empty(self.nChannels)
                }
        for ichannel in range(self.nChannels):
            mean, median, std = sigma_clipped_stats(self.origImg[ichannel], sigma=self.__sigma)
            self.__origStats['mean'] = mean
            self.__origStats['median'] = median
            self.__origStats['std'] = std

    @property
    def origStats(self):
        if self.__origStats is None:
            self.__calcOrigStats()
        return self.__origStats

    def __calcLumStats(self):
        mean, median, std = sigma_clipped_stats(self.lum, sigma=self.__sigma)
        self.__lumStats = {
                'mean': mean,
                'median': median,
                'std': std
                }

    @property
    def lumStats(self):
        if self.__updateLumStats:
            self.__calcLumStats()
            self.__updateLumStats = False
        return self.__lumStats

    def __calcStats(self):
        if self.nChannels == 2:
            self.__stats = self.lumStats
            return
        self.__stats = {
                'mean': np.empty(self.nChannels),
                'median': np.empty(self.nChannels),
                'std': np.empty(self.nChannels)
                }
        for ichannel in range(self.nChannels):
            mean, median, std = sigma_clipped_stats(self.img[ichannel], sigma=self.__sigma)
            self.__stats['mean'] = mean
            self.__stats['median'] = median
            self.__stats['std'] = std

    @property
    def stats(self):
        if self.__updateStats:
            self.__calcStats()
            self.__updateStats = False
        return self.__stats

    @property
    def nChannels(self):
        if len(self.__origImg.shape) == 2:
            return 1
        else:
            return self.__origImg.shape[0]

    def stars(self):
        if self.__stars is not None:
            return self.__stars

        if not 'fwhm' in self.__finderOptions:
            self.__finderOptions['fwhm'] = 3.0
        if not 'threshold' in self.__finderOptions:
            self.__finderOptions['threshold'] = 20.0 * self.stats['std']
        daofind = DAOStarFinder(**self.__finderOptions)
        res = daofind(self.lum - self.lumStats["median"])
        self.__stars = []

        if self.nChannels == 1:
            imgs = [self.img]
        else:
            imgs = self.img

        for ii in range(len(res['xcentroid'])):
            self.__stars.append(Star(imgs, int(round(res['ycentroid'][ii])), int(round(res['xcentroid'][ii]))))

        return self.__stars

    @property
    def starsx0(self):
        return np.array([thing.x0 for thing in self.stars()])

    @property
    def starsy0(self):
        return np.array([thing.y0 for thing in self.stars()])

    def showStars(self):
        positions = np.transpose((self.starsy0, self.starsx0))
        apertures = CircularAperture(positions, r=4.0)
        self.__plotImg(stretch=True)
        apertures.plot(color='blue', lw=1.5, alpha=0.5)
        plt.show()

    def peaks(self, threshold, box_size=31, blurSigma=10):
        img = gaussian_filter(self.lum, blurSigma)
        return find_peaks(img, threshold, box_size=box_size)

    def showPeaks(self, threshold, box_size=31, blurSigma=10, stretch=True):
        peaks = self.peaks(threshold, box_size=box_size, blurSigma=blurSigma)
        self.__plotImg(stretch=stretch)
        if peaks is not None:
            apertures = CircularAperture(np.transpose((peaks['x_peak'], peaks['y_peak'])), r=6.0)
            apertures.plot(color='blue', lw=1.5, alpha=0.5)
        plt.show()

    def findOverexposedStars(self, limit=0.95):
        peaks = self.peaks(0.0001)
        xin = peaks["x_peak"]
        yin = peaks["y_peak"]
        x = [int(round(bla)) for bla in xin]
        y = [int(round(bla)) for bla in yin]
        outx = []
        outy = []
        for ii in range(len(x)):
            if self.lum[y[ii], x[ii]] >= limit:
                outx.append(xin[ii])
                outy.append(yin[ii])
        return outx, outy

    @property
    def shape(self):
        return self.img.shape

    def addHalo(self, halo):
        if not isinstance(halo, Halo):
            raise ValueError("halo needs to be of type Halo")
        self.__img = np.add(self.img, halo.img)
        self.__setUpdate()


    @property
    def data(self):
        return self.img

    def setFinderOptions(self, options):
        self.__finderOptions = options

    def __plotImg(self, stretch=False):
        if stretch:
            img = self.stretchedImg
        else:
            img = self.img
        if len(img.shape) == 3:
            img = np.dstack([img[0], img[1], img[2]])
        print(img.shape)
        plt.imshow(img, origin='lower', interpolation='nearest')

    def showOverexposedStars(self, stretch=False):
        self.__plotImg(stretch=stretch)
        x, y = self.findOverexposedStars()
        for ii in range(len(x)):
            r = self.starExtent(x[ii], y[ii])
            aperture = CircularAperture((x[ii], y[ii]), r=r)
            aperture.plot(color='blue', lw=1.5, alpha=0.5)
        plt.show()

    def showImage(self, stretch=True):
        self.__plotImg(stretch=stretch)
        plt.show()

    def starExtent(self, x0, y0, limit=0.95):
        if self.nChannels == 1:
            radii = np.zeros(1)
            img = [self.lum]
        else:
            radii = np.zeros(3)
            img = self.img
        center = [int(round(x0)), int(round(y0))]

        for ichannel in range(len(img)):
            val = 1
            r = 0
            mask = np.empty([self.nx, self.ny], dtype=np.float32)
            while val >= limit:
                r += 1
                mask.fill(0)
                cv.circle(mask, center, r, 1, 1)
                ind = np.where(mask == 1)
                val = np.max(img[ichannel][ind])
            radii[ichannel] = r
        return np.max(radii)


    def FWHMs(self, maxredchi=0.0003, maxfwhm=15, ratiospread=1.5, maxratio=1.5):
        stars = self.stars()
        fwhms = []
        for ii in range(len(stars)):
            results = self.stars()[ii].params
            maxchi = np.max([results[bla].summary()["redchi"] for bla in range(self.nChannels)])
            those = np.array([[results[bla].best_values['sigmax'], results[bla].best_values['sigmay']] for bla in range(self.nChannels)]).flatten()
            ratios = np.array([those[x] / those[x+1] for x in range(0,self.nChannels*2,2)])
            if max(ratios) > maxratio:
                continue
            if max(ratios) / min(ratios) > ratiospread:
                continue
            foundnan = False
            for bla in those:
                if np.isnan(bla):
                    foundnan = True
            if foundnan:
                continue
            if np.min(those) > maxfwhm:
                continue
            if maxchi > maxredchi:
                continue
            fwhms.append([])
            for ichannel in range(self.nChannels):
                fwhms[-1].append(0.5 * 2.3548 * (
                        results[ichannel].best_values['sigmax'] +
                        results[ichannel].best_values['sigmay']
                            ) )
        return np.array(fwhms)

    @property
    def FWHM(self):
        return np.median(self.FWHMs())

class Star:
    def __init__(self, img, x0, y0, radius=6):
        self.__radius = radius
        self.__x0 = x0
        self.__y0 = y0

        xcenter = radius
        xstart = x0 - radius
        xstop = x0 + radius
        ycenter = radius
        ystart = y0 - radius
        ystop = y0 + radius

        nx = img[0].shape[-2]
        ny = img[0].shape[-1]

        if xstop >= nx:
            if (xstop - nx) == (radius - 1):
                xcenter -= 1
                x0 -= 1
            xstop = nx - 1
        if ystop >= ny:
            if (ystop - ny) == (radius - 1):
                ycenter -= 1
                y0 -= 1
            ystop = ny - 1
        if xstart < 0:
            xcenter = xstart + radius
            xstart = 0
        if ystart < 0:
            ycenter = ystart + radius
            ystart = 0

        x = np.arange(xstart, xstop, dtype=np.float32)
        y = np.arange(ystart, ystop, dtype=np.float32)
        y, x = np.meshgrid(y, x)
        x = x.flatten()
        y = y.flatten()

        self.__xstart = xstart
        self.__xstop = xstop
        self.__xcenter = xcenter
        self.__ystart = ystart
        self.__ystop = ystop
        self.__ycenter = ycenter
        self.__x = x
        self.__y = y
        self.__insets = []
        for ii in range(len(img)):
            self.__insets.append(img[ii][xstart:xstop,ystart:ystop])

        self.__results = None

    @property
    def x0(self):
        return self.__x0

    @property
    def y0(self):
        return self.__y0

    def fit(self):
        if self.__results is not None:
            return self.__results
        self.__results = []
        for ichannel in range(len(self.__insets)):
            z = self.__insets[0].flatten()
            model = lmfit.models.Gaussian2dModel()
            params = model.make_params(
                    amplitude = 1100.,
                    centerx = self.x0,
                    centery = self.y0,
                    sigmax = 1.0,
                    sigmay = 1.0
                    )
            self.__results.append(model.fit(z, x=self.__x, y=self.__y, params=params))

    def __plotImg(self, stretch=True):
        img = np.dstack(self.__insets)
        if stretch:
            stretcher = AutoStretch.Stretch()
            img = stretcher.stretch(img)
        plt.imshow(img)

    def show(self, stretch=True):
        self.__plotImg(stretch=stretch)
        plt.show(block=True)

    @property
    def params(self):
        if self.__results is None:
            self.fit()
        return self.__results

    def showFit(self):
        plt.title(" ".join(["".format(x.redchi) for x in self.params]))
        #plt.axis([0, self.__ystop, self.__xstart, self.__xstop])
        self.__plotImg(stretch=True)
        colors = ["red", "orange", "blue"]
        for ii in range(len(self.params)):
            p = self.params[ii]
            centerx = p.best_values['centerx'] - self.__xstart
            centery = p.best_values['centery'] - self.__ystart
            aperture = CircularAperture([centery, centerx], r=3.0)
            aperture.plot(color=colors[ii], lw=1.5, alpha=0.5)
            plt.plot([
                    centery-2.35482*p.best_values["sigmay"],
                    centery+2.35482*p.best_values["sigmay"]
                ],[centerx, centerx], color="green")
            plt.plot([centery, centery],[
                    centerx-2.35482*p.best_values["sigmax"],
                    centerx+2.35482*p.best_values["sigmax"]
                ],color="green")
        plt.show(block=True)
        plt.pause(0.1)
        plt.cla()

def isarray(thing):
    return isinstance(thing, list) or isinstance(thing, np.ndarray)

def isnumber(thing):
    return isinstance(thing, numbers.Number)

class Halo:

    def __init__(self,
            imageDimensions,
            insertPosition,
            radius,
            intensity,
            blur = 10,
            noise = 0,
            shadowSize = 0,
            vanesNumber = 0,
            vanesAngle = 0,
            vanesThickness = 50
            ):

        if not isarray(imageDimensions) or len(imageDimensions) != 2:
            raise ValueError("imageDimensions needs to be a list/array of length 2")
        for thing in imageDimensions:
            if not isnumber(thing):
                raise ValueError("imageDimensions values need to be numbers")
            if thing <= 0:
                raise ValueError("imageDimensions need to be larger than 0")

        if not isarray(insertPosition) or len(insertPosition) != 2:
            raise ValueError("insertPosition needs to be a ist/array of length 2")
        for thing in insertPosition:
            if not isnumber(thing):
                raise ValueError("insertPosition values need to be numbers")
            if thing < 0:
                raise ValueError("insertPosition need to be at least 0")

        if not isnumber(radius):
            raise ValueError("radius needs to be a number")
        if radius <= 0:
            raise ValueError("radius needs to be larger than 0")

        if isnumber(intensity):
            self.__nChannels = 1
            intensity = [intensity]
        elif isarray(intensity):
            self.__nChannels = len(intensity)
        else:
            raise ValueError("intensity needs to be a single value or a list/array")
        for thing in intensity:
            if thing <= 0:
                raise ValueError("All intensities must be larger than 0")
        if self.__nChannels != 1 and self.__nChannels != 3:
            raise ValueError("Exactly 1 or 3 channels are supported")

        if isnumber(noise):
            self.__noise = np.empty(self.__nChannels)
            self.__noise.fill(noise)
        elif isarray(noise):
            if len(noise) != self.__nChannels:
                raise ValueError("Number of noise values and channels don't match")
        else:
            raise ValueError("noise needs to be a single number of a list/array with the same length as intensity")
        for thing in self.__noise:
            if thing < 0:
                raise ValueError("noise values need to be at least 0")

        if not isnumber(shadowSize):
            raise ValueError("shadowSize needs to be a number")
        if shadowSize < 0:
            raise ValueError("shadowSize needs to be at least 0")

        if type(vanesNumber) is not int:
            raise ValueError("vanesNumber needs to be an int")
        if vanesNumber < 0:
            raise ValueError("vanesNumber needs to be at least 0")

        if not isnumber(vanesAngle):
            raise ValueError("vanesAngle needs to be a number")
        if vanesAngle < 0 or vanesAngle > 360:
            raise ValueError("vanesAngle needs to be between 0 and 360")

        if not isnumber(vanesThickness):
            raise ValueError("vanesThickness needs to be a number")
        if vanesThickness < 0:
            raise ValueError("vanesThickness needs to be at least 0")

        self.__imageDimensions = imageDimensions
        self.__insertPosition = insertPosition
        self.__radius = radius
        self.__intensity = intensity
        self.__blur = blur
        self.__shadowSize = shadowSize
        self.__vanesNumber = vanesNumber
        self.__vanesAngle = vanesAngle
        self.__vanesThickness = vanesThickness
        self.__img = None

    def __addVane(self, angle):
        x0 = self.__insertPosition[0]
        y0 = self.__insertPosition[1]
        x1 = int(round(x0 + 1.1 * self.__radius * math.cos(angle / 180. * math.pi), 0))
        y1 = int(round(y0 + 1.1 * self.__radius * math.sin(angle / 180. * math.pi), 0))
        cv.line(self.__img, [x0, y0], [x1, y1], 0, self.__vanesThickness)

    def __addVanes(self):
        angleStep = 360. / self.__vanesNumber
        for ii in range(0, self.__vanesNumber):
            angle = self.__vanesAngle + ii * angleStep
            self.__addVane(angle)

    def __addNoise(self):
        s = list(cv.split(self.__img))
        for ii in range(self.__nChannels):
            if self.__noise[ii] == 0:
                continue
            mask = (s[ii] > 0).astype(int).astype(np.float32)
            tmp = np.zeros(self.__imageDimensions, dtype=np.float32)
            tmp = cv.randn(tmp, 0., self.__noise[ii])
            tmp = cv.multiply(mask, tmp)
            s[ii] = np.add(s[ii], tmp)
            s[ii][ (s[ii] < 0) ] = 0.
        self.__img = cv.merge(s)

    def __setup_halo(self):
        center = [self.__insertPosition[0], self.__insertPosition[1]]
        self.__img = np.zeros([self.__imageDimensions[0], self.__imageDimensions[1], self.__nChannels], dtype=np.float32)
        cv.circle(self.__img, center, self.__radius, self.__intensity, -1)

        if self.__shadowSize > 0:
            cv.circle(self.__img, center, self.__shadowSize, 0, -1)

        if self.__vanesNumber > 0:
            self.__addVanes()

        if self.__blur > 0:
            self.__img = cv.blur(self.__img, [self.__blur, self.__blur])

        if (self.__noise > 0).any():
            self.__addNoise()

    @property
    def img(self):
        if self.__img is None:
            self.__setup_halo()
        return self.__img

    def save(self, fname):
        hdu = fits.PrimaryHDU(cv.split(self.img))
        hdul = fits.HDUList([hdu])
        hdul.writeto(fname, overwrite=True)

    def add(self, other):
        self.__img = np.add(self.img, other)