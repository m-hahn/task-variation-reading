

-- assumes the dictionary is zero-indexed, and outputs a one-indexed one
function readDictionaryReverse()
   local chars2Nums = {}
   io.input(dictLocation)
   t = io.read("*all")
   for line in string.gmatch(t, "[^\n]+") do
     local isInSecond = false
     local character
     local num
       for x in string.gmatch(line, "[^ ]+") do
          if isInSecond == true then
            character = x
            break
          else
            num = x
          end
          isInSecond = true
       end
       chars2Nums[character] = num+1
   end
   io.input():close()
   return chars2Nums
end





local replaceWordsByInts

function replaceWordsByInts(tree, position, dict)
   local sort = type(tree)
   if sort == "table" then
     for i=1, #tree do
       tree[i] = replaceWordsByInts(tree[i], i, dict)
     end
   elseif sort == "string" then
     if position == 1 then
        return tree
     else
        local num = math.min(dict[tree]) --assuming the dictionary is one-indexed
        if num == nil then
          num = params.vocab_size
        end
        return num
     end
   end
end

function readATreebankFile(fileName,intermediateDir, boundByMaxChar)
   local trees = {}
   if intermediateDir == nil then
      intermediateDir = ""
   end
   if boundByMaxChar == nil then
      boundByMaxChar = true
   end
   io.input(corpusDir..intermediateDir..fileName)
   t = io.read("*all")
   for line in string.gmatch(t, "[^\n]+") do
-- assume that each parse is a line, and that it starts with "( (". All of this is assumed to be legit, everything else is discarded.
"( (S (NP (NP (NNP Wong) (NNP Kang-tai)) (, ,) (NP (NP (DT a) (JJ 58-year-old) (NN ward) (NN attendant)) (PP (IN at) (NP (NML (NNP Prince) (PP (IN of) (NP (NNP Wales)))) (NNP Hospital))))) (VP (VBD became) (NP (NP (DT the) (JJ seventh) (NN medic)) (SBAR (S (VP (TO to) (VP (VB succumb) (PP (TO to) (NP (NP (DT the) (JJ Severe) (NNP Acute) (NNP Respiratory) (NNP Syndrome)) (PRN (-LRB- -LRB-) (NP (NNP SARS)) (-RRB- -RRB-)))) (SBAR (WHADVP (WRB when)) (S (NP (PRP she)) (VP (VBD died) (NP (JJ late) (NNP Saturday))))))))))) (. .)))"

       if string.sub(line, 1,3) == "( (" then

           line = string.gsub(line, "%)", " ) ")
          line = string.gsub(line, "%(", " ( ")
line = string.gsub(line, "  ", " ")
line = string.gsub(line, "  ", " ")
line = string.gsub(line, "%) %(", "),(")

line = string.gsub(line, "%( %(", "((") --have to be applied twice for ((((( etc.
line = string.gsub(line, "%) %)", "))")
line = string.gsub(line, "%( %(", "((")
line = string.gsub(line, "%) %)", "))")

line = string.gsub(line, "%) ", "),'")
line = string.gsub(line, " %(", "',(")

line = string.gsub(line, "%( ", "('")
line = string.gsub(line, " %)", "')")

line = string.gsub(line, "%(", "{")
line = string.gsub(line, "%)", "}")

line = string.gsub(line, " ", "','")

loadstring("parsetree = \""..string.sub(line, 3, string.len(line)-2))()



-- TODO simply transform it to Lua code

         parsetree = replaceWordsByInts(parsetree,1)

         table.insert(trees, parsetree)
     else
        print(line)
     end


   end
   io.input():close()
   return trees
end

