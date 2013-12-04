var test = require('tape');
var orientationScore = require('./orientation-scores.js');
var lenaColour = require('lena');
var zeros = require('zeros');
var ops = require('ndarray-ops');

var lena = zeros([lenaColour.shape[0],lenaColour.shape[1]]);
ops.addeq(lena, lenaColour.pick(null,null,0));
ops.addeq(lena, lenaColour.pick(null,null,1));
ops.addeq(lena, lenaColour.pick(null,null,2));
ops.divseq(lena, 3);

test("inversion", function(t) {
    var i, j, lenaO, lenaAccu, no = [2, 3, 8], err;
    for(i=0; i<no.length; i++) {
        lenaO = orientationScore(lena, no[i]);
        lenaAccu = zeros(lena.shape);
        for(j=0; j<no[i]; j++) {
            ops.addeq(lenaAccu, lenaO.pick(j));
        }
        ops.subeq(lenaAccu, lena);
        ops.powseq(lenaAccu, 2);
        err = Math.sqrt(ops.sum(lenaAccu)/(lena.shape[0]*lena.shape[1]));
        t.ok(err<=1e-3, "RMS error <= 1e-3 (is actually " + err + ").");
    }
    
    t.end();
});

test("flat image", function(t) {
    var i, err, img, imgO;
    img = zeros([10,30]);
    ops.assigns(img, 1);
    imgO = orientationScore(img, 6);
    for(i=0; i<6; i++) {
        ops.mulseq(imgO.pick(i), 6);
        ops.subeq(imgO.pick(i), img);
        ops.powseq(imgO.pick(i), 2);
        err = Math.sqrt(ops.sum(imgO.pick(i))/(img.shape[0]*img.shape[1]));
        t.ok(err<=1e-3, "RMS error <= 1e-3 (is actually " + err + ").");
    }
    
    t.end();
});

test("rotation invariance", function(t) {
    var i, err, img1, img2, img1O, img2O, img1Os, img2Os, numOrientations = 8;
    img1 = zeros([31,91]);
    img2 = zeros([31,91]);
    ops.assigns(img1, 1);
    ops.assigns(img2, 1);
    for(i=-3; i<=3; i++) {
        img1.set(15+i,45, 3);
        img2.set(15,45+i, 3);
    }
    img1O = orientationScore(img1, numOrientations);
    img2O = orientationScore(img2, numOrientations);
    img1Os = img1O.lo(0,0,30).hi(numOrientations,31,31);
    img2Os = img2O.lo(0,0,30).hi(numOrientations,31,31);

    for(i=0; i<numOrientations; i++) {
        ops.subeq(img1Os.pick(i), img2Os.pick((numOrientations*3/2-i)%numOrientations).transpose(1,0));
        ops.powseq(img1Os.pick(i), 2);
        err = Math.sqrt(ops.sum(img1Os.pick(i))/(img1Os.shape[1]*img1Os.shape[2]));
        t.ok(err<=1e-3, "RMS error <= 1e-3 (is actually " + err + ", " + i + "-" + ((numOrientations*3/2-i)%numOrientations) + ").");
    }
    
    t.end();
});
