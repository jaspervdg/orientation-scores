var zeros = require('zeros');
var ndarray = require('ndarray');
var fft = require('ndarray-fft');
var ops = require('ndarray-ops');

// Creates the required kernels in the frequency domain.
function makeKernels(shape, numOrientations) {
    var r, c, rd, cd, angle, i,
        rows = shape[0], cols = shape[1],
        halfRows=rows>>1, halfCols=cols>>1,
        kernels = ndarray(new Float32Array(numOrientations*h*w), [numOrientations,h,w]);
    // Assign double cones of the spectrum to each kernel, using a very simple (first order) B-spline to ensure that for each position the total comes to one.
    // TODO: Use higher order (cardinal) spline.
    for(r=0; r<rows; r++) {
        for(c=0; c<cols; c++) {
            // TODO: What if r==h-r? Should we average the results for both directions?
            rd = r<rows-r ? r : (r-rows);
            cd = c<cols-c ? c : (c-cols);
            angle = Math.atan2(rd,cd);
            angle = Math.min(Math.abs(angle),Math.PI-Math.abs(angle));
            angle *= numOrientations/Math.PI;
            for(i=0; i<numOrientations; i++) {
                kernels.set(i,r,c, Math.max(0,1-Math.abs(angle-i)));
            }
        }
    }
    // Make sure the "center" frequency is always set to 1/numOrientations, so it sums to one.
    for(i=0; i<numOrientations; i++) {
        kernels.set(i,0,0, 1/numOrientations);
    }
    // TODO: Force kernels to have bounded spatial support.
    // The easiest method is to transform them to the spatial domain, multiply them with a bell function of some sort, and then transform back.
    // If the "center" value of the bell function is one, this preserves the required properties of the orientation score kernels.
}

// Computes the orientation scores of imgarr, using numOrientations orientations.
// For now we can only handle 2D images.
// TODO: Make imgnew to have a data type that is kind of the "join" of the types of the kernels we make and the data type used for imgarr.
module.exports = function(imgarr, numOrientations) {
    assert(imgarr.shape.length == 2);
    assert(numOrientations >= 1);
    var i,
        rows = imgarr.shape[0], cols = imgarr.shape[1],
        kernels = makeKernels(imgarr.shape, numOrientations),
        imgnew = ndarray(new Float32Array(numOrientations*rows*cols), [numOrientations,rows,cols]),
        imgnewI = zeros([numOrientations,rows,cols]);
        
    // TODO: Use an FFT that can deal "natively" with real-only signals.
    for(i=0; i<numOrientations; i++) {
        ops.assign(imgnew.pick(i), imgarr);
        fft(1, imgnew.pick(i),imgnewI.pick(i));
        ops.muleq(imgnew.pick(i), kernels.pick(i));
        ops.muleq(imgnewI.pick(i), kernels.pick(i));
        fft(-1, imgnew.pick(i),imgnewI.pick(i));
    }

    return imgnew;
}
