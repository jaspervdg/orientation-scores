var orientationScore = require('./orientation-scores.js');
var lenaColour = require('lena');
var zeros = require('zeros');
var ops = require('ndarray-ops');
var savePixels = require('save-pixels');
var fs = require('fs');

var lena = zeros([lenaColour.shape[0],lenaColour.shape[1]]);
ops.addeq(lena, lenaColour.pick(null,null,0));
ops.addeq(lena, lenaColour.pick(null,null,1));
ops.addeq(lena, lenaColour.pick(null,null,2));
ops.divseq(lena, 3);

var lenaO = orientationScore(lena, 6);
ops.maxseq(lenaO, 0); // Clip output just in case.
ops.minseq(lenaO, 255);
for(i=0; i<lenaO.shape[0]; i++) {
    savePixels(lenaO.pick(i), "png").pipe(fs.createWriteStream("lena-" + i + ".png"));
}
