readFiles = {}

   if DATASET == 5 or DATASET == 12 or DATASET == 15 or DATASET == 18 then
      BOUND_FOR_TARGETS = 20
   elseif DATASET == 6 then
     BOUND_FOR_TARGETS = 4
   else
     if params.TASK == "neat-qa" then
       assert(false)
     else
       print("WARNING 11 25")
     end
   end

readFiles.PERTURB_ENTITIES = true--false

--http://stackoverflow.com/questions/4990990/lua-check-if-a-file-exists
function readFiles.fileExists(name)
   local f=io.open(name,"r")
   if f~=nil then
     io.close(f)
     return true
   else return false end
end

function readFiles.readAFile(fileName,intermediateDir, boundByMaxChar)
   local dataPointFromFile = {}
   if intermediateDir == nil then
      intermediateDir = ""
   end
   if boundByMaxChar == nil then
      boundByMaxChar = true
   end


----- DEAL WITH NONEXISTING FILES -----
   -- WARNING it might be dangerous that no error is returned when the filename is nil (which happens when the counter is beyond the end of the corpus. Here, readAFile is intended to return a dummy file consisting of "1")
   
   if fileName == nil then
      fileName = "DUMMYFILETHATSHOULDNOTEXIST"
   end
   local filePath = readDict.corpusDir..intermediateDir..fileName

--   print("76t7")
  -- print(fileName..'   '..intermediateDir..'  '..tostring((readFiles.fileExists(filePath))))

   --print(filePath)
   if(not (readFiles.fileExists(filePath))) then
     local text = {}
     table.insert(text,1)
     return text
   end
-----------------------------------------


   io.input(filePath)
   t = io.read("*all")
   local counter = 0
   for line in string.gmatch(t, "[^\n]+") do
     counter = counter+1
     lineList = {}
     for token in string.gmatch(line, "[0-9]+") do
        if boundByMaxChar then
           token = math.min(token+1, params.vocab_size)
        else
           token = token + 1
        end
        table.insert(dataPointFromFile, token)
     end
   end
   io.input():close()
   return dataPointFromFile
end



function readFiles.perturbAndNormalizeIDs(q,a,t)
   -- perturb and then normalize. For instance, first take the AnswerID->LexiconNumber mapping (which has to be created). Create a list of
   -- entities occurring in the item, and create a random bijection between this map and {1, ..., n}
   -- then go through the entities occuring in the item and 
   local entitiesOccurring = {}
   --print(readDict.chars)
   --crash()
   local maxEntity = 1
   for i=1, #t do
     local entityIDOrNil = readDict.numbersToEntityIDs[t[i]]
     if entityIDOrNil ~= nil then
       entitiesOccurring[entityIDOrNil] = 1
     end
   end
   for i=1, #q do
     local entityIDOrNil = readDict.numbersToEntityIDs[q[i]]
     if entityIDOrNil ~= nil then
       entitiesOccurring[entityIDOrNil] = 1
     end
   end

   --this is for the case when the answer file did not actually exist and a 1 was supplied
   if(readDict.numbersToEntityIDs[a[1]] == nil) then
     a[1] = readDict.entityIDToNumberIndex[1]
   end

   entitiesOccurring[readDict.numbersToEntityIDs[a[1]]] = 1
   --print(entitiesOccurring)

   -- number of entities occurring
   local numberOfEntitiesOccurring = 0
   for entity, _ in pairs(entitiesOccurring) do
     numberOfEntitiesOccurring = 1 + numberOfEntitiesOccurring
   end

   --print(#entitiesOccurring)
   
   --print(readDict.entityIDToNumberIndex)
   --print(readDict.maximumOccurringEntity)

   -- now create a random injection into [1,...,readDict.maximumOccurringEntity]
   -- goes from entity IDs to entity IDs
   local randomInjectionIntoEntityIDs = {}

   local hasBeenSelectedTargetIDs = {}

   local boundOnTargets = math.max(BOUND_FOR_TARGETS, numberOfEntitiesOccurring)
   

   -- create a random ID

   if true then
     for occurringID, _ in pairs(entitiesOccurring) do
       local targetID
       repeat
         targetID = math.random(boundOnTargets)
       until(hasBeenSelectedTargetIDs[targetID] == nil)
       randomInjectionIntoEntityIDs[occurringID] = targetID
       hasBeenSelectedTargetIDs[targetID] = 1
     end
   else
   -- ordered by appearance
     local firstAppearanceListOfEntities = {}
     local numberOfEntitiesProcessedSoFar = 0
     local processEntity = function(tokenNumber)
                              local entityOrNil = readDict.numbersToEntityIDs[tokenNumber] 
                              if entityOrNil ~= nil then
                                 if firstAppearanceListOfEntities[entityOrNil] == nil then
                                    firstAppearanceListOfEntities[entityOrNil] = numberOfEntitiesProcessedSoFar+1
                                    numberOfEntitiesProcessedSoFar = numberOfEntitiesProcessedSoFar + 1
                                 end
                              end
                           end
     for tokenNumber in t do
        processEntity(tokenNumber)
        -- TODO do the same for q, a
    -- for occurringID, _ in pairs(entitiesOccurring) do
     end
   end

--   print("444")
  -- print(q)
   for position, tokenNumber in pairs(q) do

      local entityIDOrNil = readDict.numbersToEntityIDs[tokenNumber]
      if entityIDOrNil ~= nil then
         --print(entityIDOrNil)
         --print(randomInjectionIntoEntityIDs)
         --print(readDict.entityIDToNumberIndex)
         q[position] = readDict.entityIDToNumberIndex[randomInjectionIntoEntityIDs[entityIDOrNil ]]
--         print("9743")
  --       print(q[position])
      end
   end
   --print("5555")
   --print(q)
   for position, tokenNumber in pairs(a) do
      local entityIDOrNil = readDict.numbersToEntityIDs[tokenNumber]
      if entityIDOrNil ~= nil then
    --      print(entityIDOrNil)
      --   print(randomInjectionIntoEntityIDs)
        
         a[position] = readDict.entityIDToNumberIndex[randomInjectionIntoEntityIDs[entityIDOrNil ]]
        -- print("8643")
         --print(a[position])
      end
   end
   for position, tokenNumber in pairs(t) do
      local entityIDOrNil = readDict.numbersToEntityIDs[tokenNumber]


      if entityIDOrNil ~= nil then
--      print("--")
  --    print(tokenNumber)
    --  print(readDict.numbersToEntityIDs[tokenNumber])
      --print(randomInjectionIntoEntityIDs[readDict.numbersToEntityIDs[tokenNumber]])
      --print(readDict.entityIDToNumberIndex[randomInjectionIntoEntityIDs[readDict.numbersToEntityIDs[tokenNumber]]])




         t[position] = readDict.entityIDToNumberIndex[randomInjectionIntoEntityIDs[entityIDOrNil ]]
      end
   end
--   print("6557")
  -- print(q)
--   print(a)
  -- print(t)
   --print(a)
   --print(readDict.entityIDToNumberIndex)
   --crash()
-- print("2069")
 --print(randomInjectionIntoEntityIDs)
end



function readFiles.readNextQAItem(batchIndex)
    ::getNext::
    readChunks.corpusReading.currentFile = readChunks.corpusReading.currentFile + 1
    local fileName = readChunks.files[readChunks.corpusReading.currentFile]
    if fileName == nil then -- at the end of a dataset, this will become nil. just return an empty datapoint, main_attention will stop running soon after (it might go through a whole set of bins, though)
     return {question = {}, answer = {}, text = {}, baseline=0}
    end
    local sepIndex
    if DATASET == 12 then
       sepIndex = nil
    else
       sepIndex = string.find(fileName,"\t")
    end
    local baseline
    if sepIndex == nil then
      baseline = 0.0
    else
      baseline = tonumber(string.sub(fileName, sepIndex))
      fileName = string.sub(fileName, 1, sepIndex-1)
    end
    local q = readFiles.readAFile(fileName, "/questions/", true)
    local a = readFiles.readAFile(fileName, "/answers/", false)
    local t = readFiles.readAFile(fileName, "/texts/", true)


--    print("468")
--    print(q)
--print(a)
--print(t)
--    print("44")
  --  print(a)
-- some deepmind-corpus-specific things
-- NOTE the next three lines might be sensitive to subtle changes in the input format
    if #a > 1 then
       a = {a[2]}
    end
--    print("print id 51")
  --  print(a[1])
  --
  --
  --
    
    if readFiles.PERTURB_ENTITIES then
       readFiles.perturbAndNormalizeIDs(q,a,t)
    end
    ---- now here, a is replaced
    
    a[1] = readDict.numbersToEntityIDs[a[1]]

--    print(readDict.numbersToEntityIDs)
  --  print("NEW ANSWER "..a[1])
    --print(q)
--    print(t)
  --  print("24511")
    -- this can happen if the answer file did not actually exist
--    if(a[1] == nil) then
  --    a[1] = 1
    --end


    --print(a[1])

-- TODO NOTE goto could lead to crash at the end of the corpus, since a new eleemnt will be retrieved without asking hasNext
    if #q > MAX_LENGTH_Q_FOR_QA then
--       print("print id 476")
  --     print(#q)
    end
    if #t > MAX_LENGTH_T_FOR_QA then
--       print("print id 480")
  --     print(#t)
    end

    --print("66")
    --print({question = q, answer = a, text = t})

--    print("111   ")
  --  print(q)
    --print(a)
  --  print(t)
    return {question = q, answer = a, text = t, baseline=baseline}
end





-- NOTE THAT THE FILES COME FROM PYTHON AND WILL BE ZERO-INDEXED
function readFiles.readCorpus() --as a double storage
   io.input(corpus_name..".num.b")
   t = io.read("*all")
   local counter = 0
   for line in string.gmatch(t, "[^\n]+") do
     counter = counter+1
     lineList = {}
     table.insert(readChunks.corpus, lineList)
     for token in string.gmatch(line, "[0-9]+") do
        table.insert(lineList, math.min(token+1.0, params.vocab_size))
     end

    --print(line)

   end
   --corpus = torch.Tensor(corpusList)
   io.input():close()


   io.input(corpus_name..".charnums.txt")
   t = io.read("*all")

   for line in string.gmatch(t, "[^\n]+") do
     table.insert(readFiles.chars, line)
   end
   
   if params.vocab_size > #readFiles.chars then
      print(#readFiles.chars)
      crash()
   end
   io.input():close()
--print(chars)

end
