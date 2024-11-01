# Public defense

See also: 3blue1brown manim tutorial!
See also: manim slides! https://github.com/jeertmans/manim-slides

## Preamble: 1m00s
Voor we beginnen een kleine dienstmededeling: de presentatie gaat volledig in het Nederlands zijn,
maar de slides zijn in het Engels, zodat de Engelstaligen onder ons ook kunnen volgen.
Ik ga dit nu nog even meedelen aan de Engelstaligen, en dan kunnen we beginnen.
Hello everyone, I have some sad news: I decided to do this presentation entirely in Dutch.
I know, this is very disappointing, but please understand my situation:
today I have the opportunity to talk to my friends and family about my work for *40 minutes*,
and they *can't run away.*
This is a once in a lifetime opportunity, so please forgive me for taking it.
That being said, the slides are in English and I have added subtitles wherever possible,
so you should be able to follow using the slides.

## Introductie

### Intro part 1 (1m45s)
Om te beginnen, moeten we even terug naar iedereen zijn/haar favoriete plek en meest dierbare herinnering:
de wiskundeles in het derde middelbaar.
Ik voel de spanning al een beetje toenemen in de zaal, maar er is geen reden om bang te zijn.
Ik heb namelijk gemerkt dat de meeste mensen die een trauma hebben aan wiskunde
eigenlijk vooral een trauma hebben aan de wiskunde*toets.*
Vandaag is er geen toets, dus ook geen reden om bang te zijn.

We beginnen met het concept *functie.*
Een echte informaticus denkt in termen van in- en output,
en we kunnen een functie ook zo bekijken:
een soort "machine" die een invoer neemt en een uitvoer teruggeeft.

Onze "machine" heeft ook een soort "programmacode" of "instructies"
die duidelijk maken wat de machine met zijn invoer moet doen.
We schrijven die programmacode in een vergelijking:
aan de linkerkant hebben we f van x, dus f is onze machine en x is de invoer,
en aan de rechterkant schrijven we wat we willen dat f doet met x.
In dit geval gaat f het getal x vermenigvuldigen met 2 en er dan 1 van aftrekken.

We kunnen dit eens testen: als we het getal 3 meegeven met f,
dan krijgen we inderdaad 2 maal 3 plus 1, en dat is 7.
We kunnen hetzelfde doen met -1, en dan krijgen we 2 maal -1 plus 1, en dat is -3.

We kunnen daar een grafiekje van maken.
Hier zien we onze twee punten terug, die komen elk overeen met een invoer (links-rechts)
en de bijhorende uitvoer (de hoogte).
Als we dit herhalen voor alle getallen, krijgen we deze lijn te zien.

### Intro part 2
Laten we nog eens kijken naar onze formule.
Er zijn twee getallen die onze formule eigenlijk vastleggen: die 2 en die -1.
Als we een beetje prutsen aan die 2, dan zien we dat de richting van onze lijn verandert.
Als we prutsen aan die -1, dan gaat onze lijn naar boven en beneden.
Die 2 en die -1, we noemen die ook *parameters.*

Machine learning, of machinaal leren, is eigenlijk niets anders dan de computer 
automatisch laten zoeken naar een goeie waarde van die parameters
zodat onze functie voor iets nuttigs gebruikt kan worden.
Wat bedoel ik daarmee?
Stel dat we bijvoorbeeld op basis van de temperatuur willen kunnen voorspellen 
hoeveel energie we verbruiken in ons huis.
Als het buiten koud is, gaan we waarschijnlijk meer energie nodig hebben om te verwarmen.

Wat we kunnen doen, is op een aantal dagen meten hoe warm het buiten is,
dat is onze X-as,
en dan meten hoeveel energie we hebben verbruikt.
Hier heb ik dat gedaan voor 6 dagen, dus we hebben 6 punten op onze grafiek.




- Lineaire functie in 1 veranderlijke
  - De formule vat samen wat de functie doet. Dit zijn de "instructies" van de machine
  - De formule heeft 2 parameters die we kunnen veranderen (slope, intercept)
  - Machine learning = automatisch zoeken naar de parameters zodat de machine de gegeven data
    zo goed mogelijk kan benaderen
    - Hoe definieren we "zo goed mogelijk"? We krijgen een getal dat de "fout" (loss) weergeeft,
      hoe kleiner dit getal hoe beter.
  - Demonstratie: van random parameters naar mooie fit op lineaire data
- Wat als we 2 veranderlijken hebben?
  - Voorbeeld van surface plot (lineair), dit lukt nog, maar wordt al moeilijker
- Wat als we meerdere veranderlijken hebben (3 veranderlijken, 1 kwadratisch)?
  - Kunnen nu niet meer plotten maar we kunnen wel nog kijken naar de formule.
  - Door te kijken naar de formule kunnen we zien wat onze machine "doet"
    - Bvb feature A heeft precies niet zo veel invloed als B
- Een "echt" machine learning model, ook ChatGPT,
  ook de modellen die uit een foto kunnen herkennen of er een persoon op staat
  en wie die persoon is, is ook gewoon een functie met een bepaalde formule!
  - De veranderlijken zijn bvb de pixels in een foto (3 getallen per pixel)
  - De outputs zijn bvb "1" als de foto een persoon bevat en "0" als de foto geen persoon bevat
  - => Het doet er niet toe wat de data precies is,
    we kunnen altijd doen alsof het gewoon een lijst van getallen is.
- We keren even terug naar 2d plots
  - Wat als de data wat complexer is (kwadratisch)?
    - Lineaire fit werkt nog in principe, maar benadering is niet goed
    - Parameter toevoegen helpt
    - We kunnen nog altijd de formule gebruiken als een beschrijving van de "programmacode"
    - Waarom x^2 toevoegen en niet x^3, de sinus van x, of nog iets anders?
      - Het is de taak van de programmeur om de goeie "vorm" (model) te kiezen voor de data
  - Wat als de data nog complexer wordt? (piecewise combinatie van functies)
    - We voegen nog parameters toe, en het werkt terug.
    - Laten we kijken naar de programmacode...
      - Veel te veel parameters.
      - Dit zijn $n$ parameters. Resnet18, een redelijk klein model voor computervisie, heeft $N$ parameters!
        - Enkele fun facts over hoe gigantisch de "programmacode" is van resnet18.
- Conclusie: we kunnen een "echt" ML model enkel beschouwen als een "black box".
- Hoe kunnen we toch nog inzicht krijgen?
  - Door te kijken naar hoe de black box zich gedraagt met verschillende inputs
  - *Attribution-based explanation:* we produceren een score voor iedere invoerveranderlijke
    - Die score zegt hoe belangrijk iedere veranderlijke was voor het model in deze predictie
    - Als de invoer een foto is, kunnen we een heatmap tonen
    - Als de invoer een simpele tabel is, kunnen we een barplot tonen
- Er bestaan enorm veel methoden om "attribution-based explanations" te genereren
  - Allemaal gebaseerd op speciale wiskundige technieken waar we nu niet op hoeven ingaan
- => De grote onderzoeksvraag aan de start van mijn doctoraat:
     **hoe kunnen we weten welke methode de beste is voor een bepaalde toepassing?**

## Benchmark
- Eerste grote experiment.
- Er bestaan enorm veel manieren om een "heatmap" te maken voor een bepaalde predictie
  - Allemaal gebaseerd op speciale wiskundige technieken waar we nu niet op hoeven ingaan
  - Allemaal ontworpen om te tonen *welke regio's in de foto het belangrijkst waren voor die ene predictie.*
    - Waarom is dat nuttig? Zo kunnen we zien of het model echt geleerd heeft wat we willen dat het leert.
    - Husky voorbeeld/meetlat voorbeeld.
  - In veel gevallen genereren ze andere heatmaps.
    - Probleem: welke is nu de juiste?
- Simpele metriek: als we de 10% "belangrijkste" pixels vervangen door een grijs vierkant, verandert de predictie dan?
- Er bestaan enorm veel andere manieren om de kwaliteit van een heatmap te meten
  - Allemaal gebaseerd op speciale wiskundige technieken waar we nu niet op hoeven ingaan
  - Allemaal ontworpen om te meten *hoe "correct" de heatmap is:* zijn de "belangrijkste" regio's ook echt belangrijk?
- Simpele oefening:
  - We verzamelen een aantal datasets met afbeeldingen
  - We verzamelen een aantal methodes voor het genereren van heatmaps
  - We verzamelen een aantal kwaliteitsmetrieken voor heatmaps
  - We meten de kwaliteit van de verschillende heatmaps met de verschillende metrieken
- Resultaat:
  - "Beste" methode hangt af van de dataset...
  - ... maar ook van de metriek!
    - Conclusie: *er is niet 1 enkele maat van "correctheid".* 
      De metrieken meten toch nog onderling verschillende eigenschappen van de heatmaps.

## Removal-based attribution methods
- Stel: tabular model dat risico op diabetes moet bepalen obv lengte, leeftijd, geslacht, bloeddruk.
- We willen schatten hoe belangrijk iedere veranderlijke is.
  - De eerste vraag die we moeten beantwoorden is: voor wat?
    - bv: hoe gevoelig is het model aan iedere veranderlijke?
    - of: hoe nuttig is iedere veranderlijke voor de accuraatheid van ons model?
    - of: voor een bepaalde patient en predictie, hoe belangrijk is iedere veranderlijke voor die patient?
  - **Ingredient 1** van een verklaring: het **doelwit.**
- Stel nu: we hebben 1 patient met een predictie, en we willen weten hoe belangrijk de variabelen zijn voor die predictie.
  - Ons "doelwit" is gekozen.
  - Stel dat we willen weten hoe belangrijk "leeftijd" was.
    - Het liefst zouden we vragen aan het model: *Als we leeftijd niet kenden, wat zou dan je predictie zijn?*
      - Probleem: we moeten een waarde meegeven aan de functie.
      - Kunnen we een "lege waarde" meegeven? Op 0 zetten?
        - Probleem: dit is niet een "lege waarde", dit is "een pasgeboren baby"... Er zitten wss geen mensen met leeftijd 0 in de data
      - We kunnen alle waarden voor "leeftijd" eens invullen en kijken naar de gemiddelde uitvoer?
        - Probleem: dan krijgen we nog steeds rare combinaties, bvb 12-jarig kind met cholesterol van een 40-jarige *(todo beter voorbeeld?)*
      - Er zijn enorm veel manieren om een veranderlijke te "verwijderen", elk met voor- en nadelen.
      - **Ingredient 2** van een verklaring: de manier om variabelen te **verwijderen.**
- Stel nu: ons model is heel simpel, het voorspelt een verhoogde kans als x1 > a OF x2 > b.
  - Wat gaat onze simpele methode doen? Geen enkele variabele zal belangrijk zijn!
  - We mogen niet alleen kijken naar iedere variabele apart, maar we moeten ook kijken naar verzamelingen.
    - Dit is een voorbeeld van *interactie:* het effect van de twee variabelen samen is anders dan de som van het effect van de aparte veranderlijken.
  - Voor 4 variabelen hebben we 16 mogelijke deelverzamelingen
  - Hoe brengen we dit terug naar 1 score per veranderlijke?
  - **Ingredient 3** van een verklaring: de manier om de output van groepen van veranderlijken **terug samen te voegen** in 1 score per veranderlijke.
- We kunnen dit formeler maken.
  - Doelwit: Phi(f)
  - Verwijderen: P_S(f)
  - Aggregeren: alpha
- **Contributie:** we kunnen heel veel bestaande technieken onder deze noemer brengen
  - Dit maakt verschillende methodes plots makkelijker vergelijkbaar
    - Bvb methode A en B verschillen enkel in hoe ze veranderlijken verwijderen

## Functional decomposition
- Gegeven simpele functie (polynomial) 2x1 + 4x2 - 3x1x2 + x3 + 5
  - We kunnen deze functie schrijven als een som van functies die elk afhangen van een deelverzameling variabelen
    - De componenten met 1 variabele zijn "directe effecten", de componenten met meerdere zijn "interacties"
    - Voorbeeld van onze functie op een bepaald punt: we kunnen ieder component zien als een "bijdrage" van een (groep) veranderlijke(n)
  - Dit is een *additieve decompositie* (decompositie: we splitsen de functie in deeltjes, additief: het is een som)
    - De additieve decompositie is niet uniek: we kunnen termen naar boven verplaatsen
    - Maar: we mogen termen niet "naar beneden" verplaatsen
- Wat blijkt: iedere geldige additieve decompositie komt precies overeen met een manier van verwijderen P_S(f)
  - Dus: iedere attributiemethode uit ons framework kunnen we verbinden met een additieve decompositie
  - We kunnen "manier van verwijderen" vervangen door "manier van functie ontbinden"

## PDD-SHAP
- Wat is daar nu het praktisch nut van?
=> SHAP is een populaire methode: verwijdert alle mogelijke deelverzamelingen en aggregeert op een ingewikkelde manier.
Gebaseerd op speltheorie, theorie uit 1970(?) die oorspronkelijk ontwikkeld was om een bepaalde uitkomst van samenwerking eerlijk te verdelen onder spelers.

Maar: via framework kunnen we het samenvatten in 3 keuzes:
- doelwit = functie zelf, de scores zeggen wat de invloed is van iedere veranderlijke op de uitvoer van het model in x
- verwijderen = gemiddelde output over marginal distribution
- aggregeren = iedere interactie wordt gelijk verdeeld onder de leden

Computationele kost: aantal deelverzamelingen * aantal datapunten, super duur.

Als we de decompositie zouden hebben, kunnen we het gewoon berekenen als gewogen som van componenten.

Dit is het idee van PDD-SHAP: maak een nieuw model voor iedere component!

Te veel componenten? => factor sparsity

Results: R2, speed.


## Conclusie
- Benchmark => geen universele maat van "kwaliteit"
- Decompositie => we kunnen verschillende methodes vergelijken met elkaar door ze onder hetzelfde raamwerk te brengen
- PDD-SHAP => we kunnen dit raamwerk gebruiken om bestaande verklaringen sneller te berekenen







