from SPARQLWrapper import SPARQLWrapper, JSON
import json

paras = {}
with open('top5.json', 'r') as json_file:  
    data = json.load(json_file)
    image_namelist = '../train2014_filenames.txt'
    with open(image_namelist, 'r') as infile:
        for image_name in infile:
            li = data[image_name.strip('\n')]
            para = ""
            for l in li:
                parsed_l = '"'+l+'"'
                sparql = SPARQLWrapper("http://dbpedia.org/sparql")
                sparql.setQuery("""
                    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                    SELECT ?comment
                    WHERE { ?entry rdfs:label """ +parsed_l+"""@en.
                            ?entry rdfs:comment ?comment }
                    """)

                sparql.setReturnFormat(JSON)
                results = sparql.query().convert()

                for result in results["results"]["bindings"]:
                    if result["comment"]["xml:lang"] == "en":
                        para += str(result["comment"]["value"]) + " "
                        
            paras[image_name.strip('\n')] = para

with open('top5para.json', 'w') as out_para:
        json.dump(paras, out_para)
