from SPARQLWrapper import SPARQLWrapper, N3
from rdflib import Graph

def query_database():
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")

    queryString = "select distinct ?Concept where {[] a ?Concept} "

    #sparql.setQuery("""
    #    DESCRIBE <http://dbpedia.org/resource/Asturias>
    #""")

    sparql.setQuery(queryString)

    sparql.setReturnFormat(N3) # JSON
    try :
        results = sparql.query().convert()
        # ret is a stream with the results in XML, see <http://www.w3.org/TR/rdf-sparql-XMLres/>
    except :
       print("exception!!")
       exit(0)
    g = Graph()
    g.parse(data=results, format="n3")
    out = g.serialize(format='n3')
    print(type(out))
    print(out)

def describe_data(data):
    queryString = "DESCRIBE ?Concept where {[] a ?Concept} "


if __name__ == "__main__":
    data = query_database()
    describe_data(data)
