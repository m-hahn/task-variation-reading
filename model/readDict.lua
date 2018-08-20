readDict = {}


readDict.chars = {}

function readDict.readDictionary()
   io.input(readDict.dictLocation)
   t = io.read("*all")
   for line in string.gmatch(t, "[^\n]+") do
     local isInSecond = false
     local character
       for x in string.gmatch(line, "[^ ]+") do
          character = x
          if isInSecond == true then
            break
          end
          isInSecond = true
       end
       table.insert(readDict.chars, character)
   end
   io.input():close()
   assert(params.vocab_size <= #readDict.chars, tostring(#readDict.chars))
end


-- essentially, the entityNumber/entityID should be used in the softmax
function readDict.createNumbersToEntityIDsIndex()
    readDict.numbersToEntityIDs = {}
    readDict.entityIDToNumberIndex = {}
    readDict.maximumOccurringEntity = 0
    for num, char in pairs(readDict.chars) do
        if char:sub(1,7) == "@entity" then
            entityNumber = char:sub(8)+1 --better add 1 here because the entities start at 0
            readDict.numbersToEntityIDs[num+0] = entityNumber
            readDict.entityIDToNumberIndex[entityNumber] = num
            readDict.maximumOccurringEntity = math.max(readDict.maximumOccurringEntity, entityNumber)
        end
    end
    assert(NUMBER_OF_ANSWER_OPTIONS ~= nil)
    assert(readDict.maximumOccurringEntity ~= nil)

    if(NUMBER_OF_ANSWER_OPTIONS<readDict.maximumOccurringEntity) then
       print("--")
       print( (NUMBER_OF_ANSWER_OPTIONS))
       print( readDict.maximumOccurringEntity)
       crash()
    end
end



function readDict.setToPretrainedEmbeddings(tensor)
   print("Starting to read pretrained embeddings")
   local embPaths = {}
   embPaths["/afs/cs.stanford.edu/u/mhahn/num2CharsNoEntities-deepmind-joint"] = "/u/scr/mhahn/glove-100d-joint-noentities"
   embPaths["/u/scr/mhahn/num2CharsNoEntities-cnn"] = "/u/scr/mhahn/glove-100d-cnn-entities"
   embPaths["/u/scr/mhahn/testdata/inference/num2Chars"] = "/u/scr/mhahn/glove-100d-cnn-entities"

   local embPath = embPaths[readDict.dictLocation]
   local pretrainedEmbeddings = unpack(torch.load(embPath, "binary"))
   assert(params.embeddings_dimensionality == pretrainedEmbeddings:size(2))
   assert(params.vocab_size <= pretrainedEmbeddings:size(1))
   pretrainedEmbeddings = pretrainedEmbeddings:narrow(1,1,params.vocab_size)
   assert(tensor:size(2) == pretrainedEmbeddings:size(2))
   if tensor:size(1) == params.vocab_size + 1 then
     local maskZero = torch.FloatTensor(1,tensor:size(2))
     pretrainedEmbeddings = torch.cat(maskZero,pretrainedEmbeddings,1)
   else
     assert(tensor:size(1) == params.vocab_size)
   end
   assert(tensor:size(1) == pretrainedEmbeddings:size(1))
   assert(tensor:size(2) == pretrainedEmbeddings:size(2))
   assert(tensor:dim() == 2)
   assert(pretrainedEmbeddings:dim() == 2)
   tensor:copy(pretrainedEmbeddings)
   print("Read pretrained embeddings from "..embPath) 
end

-- assumes readDict.chars has already been filled
-- call
--              readDict.createEmbeddingLayer("/u/scr/mhahn/glove-100d-cnn-entities")
-- after the dictionary has been read in
-- creates a binary file with embeddings at the indicated path
function readDict.createEmbeddingLayer(outFilePath)
  print("Loading embeddings")
   local Glove = require 'glove'
   print("Transferring embeddings")
   local embeddingTensor = torch.FloatTensor(#readDict.chars, Glove.M:size(2))
   embeddingTensor:uniform(-params.init_weight,params.init_weight)
print(#readDict.chars)
   for i=1,#readDict.chars do
--      print(i)
      local id = Glove.w2vvocab[readDict.chars[i]]
      if id ~= nil then
        embeddingTensor[i] = Glove.M[id]
      else
        print(i.."  5716 Did not find "..(readDict.chars[i]))
      end
   end
   torch.save(outFilePath, {embeddingTensor}, "binary")

end


function readDict.word2Num(word)
  for i=1,#readDict.chars do
   if readDict.chars[i] == word then
     return i
   end
  end
  return 0
end
