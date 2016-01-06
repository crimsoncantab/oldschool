<?
    // require code for creating practice times
    require_once("inc/generator.inc");
?>

<!DOCTYPE html 
     PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN"
               "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">

<html xmlns="http://www.w3.org/1999/xhtml">
  <head>
    <title>Practice Times</title>
  </head>
  <body>
    <table border="0" width="100%">
      <tr>
        <td valign="top">
          <form action="email.php" method="post">
            <div>
              <textarea rows="20" cols="50" name="message"><?
                //display practice times in textbox
                print($message);
              ?></textarea>
            </div>
            <div>
              E-mail to:
              <input type="text" name="to" />
            </div>
            <div>
              Subject:
              <input type="text" name="subject" value="Practice Times" />
            </div>
            <div>
              <input type="submit" value="E-mail" />
            </div>
          </form>
          <div>
            <a href="logout.php">Log Out</a>
          </div>
        </td>
        <td>
        <div align="center">
        <table border="1">
          <tr>
            <th>Ports</th>
            <th>Availability</th>
          </tr>
          <?
                //imports port athletes from database to be listed in table
              $sql = mysql_query("SELECT * FROM " . $_SESSION["username"] . " WHERE pref='p'");
              while ($row = mysql_fetch_array($sql)) {
                  print("<tr>");
                  print("<td>" . $row["athlete"] . "</td>");
                  print("<td>");
                  $pre = $row["pretime"];
                  $prem = ($pre - floor($pre)) * 60;
                  if ($prem == 0) $prem = "";
                  else $prem = ":" . $prem;
                  $post = $row["posttime"];
                  $postm = ($post - floor($post)) * 60;
                  if ($postm == 0) $postm = "";
                  else $postm = ":" . $postm;
                  if ($pre < 1) $pre += 12;
                  if ($post < 1) $post += 12;
                  print(floor($pre) . $prem . " - " .  floor($post) . $postm . "</td>");
                  print("</tr>");
              }
          ?>
        </table>
        </div>
        </td>
        <td>
        <div align="center">
        <table border="1">
          <tr>
            <th>Starboards</th>
            <th>Availability</th>
          </tr>
          <?
                //imports starboard athletes from database to be listed in table
              $sql = mysql_query("SELECT * FROM " . $_SESSION["username"] . " WHERE pref='s'");
              while ($row = mysql_fetch_array($sql)) {
                  print("<tr>");
                  print("<td>" . $row["athlete"] . "</td>");
                  print("<td>");
                  $pre = $row["pretime"];
                  $prem = ($pre - floor($pre)) * 60;
                  if ($prem == 0) $prem = "";
                  else $prem = ":" . $prem;
                  $post = $row["posttime"];
                  $postm = ($post - floor($post)) * 60;
                  if ($postm == 0) $postm = "";
                  else $postm = ":" . $postm;
                  if ($pre < 1) $pre += 12;
                  if ($post < 1) $post += 12;
                  print(floor($pre) . $prem . " - " .  floor($post) . $postm . "</td>");
                  print("</tr>");
              }
          ?>
        </table>
        </div>
        </td>
      </tr>
      <tr><td></td>
        <td>
        <div align="center">
        <table border="1">
          <tr>
            <th>Coxswains</th>
            <th>Availability</th>
          </tr>
          <?
                //imports coxswain athletes from database to be listed in table
              $sql = mysql_query("SELECT * FROM " . $_SESSION["username"] . " WHERE pref='c'");
              while ($row = mysql_fetch_array($sql)) {
                  print("<tr>");
                  print("<td>" . $row["athlete"] . "</td>");
                  print("<td>");
                  $pre = $row["pretime"];
                  $prem = ($pre - floor($pre)) * 60;
                  if ($prem == 0) $prem = "";
                  else $prem = ":" . $prem;
                  $post = $row["posttime"];
                  $postm = ($post - floor($post)) * 60;
                  if ($postm == 0) $postm = "";
                  else $postm = ":" . $postm;
                  if ($pre < 1) $pre += 12;
                  if ($post < 1) $post += 12;
                  print(floor($pre) . $prem . " - " .  floor($post) . $postm . "</td>");
                  print("</tr>");
              }
          ?>
        </table>
        </div>
        </td>  
    </table>
  </body>
</html>
