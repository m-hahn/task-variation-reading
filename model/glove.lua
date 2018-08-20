-- adapted from https://github.com/rotmanmi/glove.torch/blob/master/glove.lua

assert(false)

torch.setdefaulttensortype('torch.FloatTensor')

opt = {
	binfilename = '/scr/nlp/data/wordvectors/en/glove/vectors_100d.txt',
	outfilename = '/u/scr/mhahn/glove-100d.t7'
}
local GloVe = {}
if not paths.filep(opt.outfilename) then
	GloVe = require('importEmbeddings.lua')
else
	GloVe = torch.load(opt.outfilename)
	print('Done reading GloVe data.')
end


GloVe.distance = function (self,vec,k)
	local k = k or 1	
	--self.zeros = self.zeros or torch.zeros(self.M:size(1));
	local norm = vec:norm(2)
	vec:div(norm)
	local distances = torch.mv(self.M ,vec)
	distances , oldindex = torch.sort(distances,1,true)
	local returnwords = {}
	local returndistances = {}
	for i = 1,k do
		table.insert(returnwords, self.v2wvocab[oldindex[i]])
		table.insert(returndistances, distances[i])
	end
	return {returndistances, returnwords}
end

GloVe.word2vec = function (self,word,throwerror)
   local throwerror = throwerror or false
   local ind = self.w2vvocab[word]
   if throwerror then
		assert(ind ~= nil, 'Word does not exist in the dictionary!')
   end
	if ind == nil then
		ind = self.w2vvocab['UNK']
	end
   return self.M[ind]
end

return GloVe




