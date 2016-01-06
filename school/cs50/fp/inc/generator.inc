<?
    // require common code
    require_once("common.inc");

    // require athlete and practice classes
    require_once("classes.inc");

    //gets variable information on constraints
    $interim = $_GET["interim"];
    $maxAthletes = $_GET["boatmax"] * 8;
    $side = $_GET["side"];
    $minAthletes = 8;
    if ($_GET["size"] == "true") $minAthletes = 4;


    //create list of athletes
    $Alist = array();

    //create list of practices
    $Plist = array();

    //import athletes from database, starting with latest pretimes
    for ($i = 7; $i >= 0; $i -= .25) {
        $sql = mysql_query("SELECT * FROM " . $_SESSION["username"] . " WHERE pretime=" . $i);
        while($row = mysql_fetch_array($sql))
            $Alist[] = new Athlete($row["athlete"], $row["pref"], $row["pretime"], $row["posttime"]);
    }

    //while number of players is large enough to fill a boat, create a practice group
    $time = (double)$_GET["lateh"] + (double)$_GET["latem"] / 60;
    if ($time > 7 - $_GET["length"])
        $time = 7 - $_GET["length"];

    //create practice times
    while($time > 0) {
        $numAvailable = 0;
        foreach ($Alist as $tempAthlete)
            if ($tempAthlete->isAvailable($time) && $tempAthlete->isUnassigned())
                $numAvailable++;
        if ($numAvailable >= $minAthletes) {
            $tempPractice = new Practice($time);
            $tempPractice->fillPractice($Alist, $minAthletes, $maxAthletes, $side);
            $Plist[] = $tempPractice;
            $time -= $interim;
        }
        else
            $time -= .25;
    }
    
    //reverse times to be listed chronologically
    $Plist = array_reverse($Plist);

    //append printed practice times to a message string
    foreach ($Plist as $tempPractice)
        if(count($tempPractice->athletes) != 0)
            $message .= $tempPractice->printPractice() . "\n\n";
    
    //appends unassigned players to message string    
    $message .= "OYO - ";

    $comma = 0;
    foreach ($Alist as $tempAthlete) {
        if ($tempAthlete->practice == 7) {
            if ($comma == 1) $message .= ", ";
            $message .= $tempAthlete->athlete;
            if ($comma == 0) $comma++;
        }
    }
?>
