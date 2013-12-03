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
        lenaO = orientationScore(lena,no[i]);
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
