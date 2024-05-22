# test image file
# make sure we can see an image
def convert(img_file, label_file, txt_file, n_images):
  print("\nOpening binary pixels and labels files ")
  lbl_f = open(label_file, "rb")   # MNIST has labels (digits)
  img_f = open(img_file, "rb")     # and pixel vals separate
  print("Opening destination text file ")
  txt_f = open(txt_file, "w")      # output file to write to

  print("Discarding binary pixel and label files headers ")
  img_f.read(16)   # discard header info
  lbl_f.read(8)    # discard header info

  print("\nReading binary files, writing to text file ")
  print("Format: 784 pixel vals then label val, tab delimited ")
  for i in range(n_images):   # number images requested
    image_pixels = []
    lbl = ord(lbl_f.read(1))  # get label (unicode, one byte) 
    for j in range(784):  # get 784 vals from the image file
      image_pixels.append(ord(img_f.read(1)))
    txt_f.write(f"<h4>{lbl}</h4>")
    txt_f.write("<table cellspacing=0 cellpadding=0>")
    for y in range(28):
      txt_f.write("<tr>")
      for x in range(28):
        color = f"rgb({image_pixels[y*28+x]},{image_pixels[y*28+x]},{image_pixels[y*28+x]})"
        txt_f.write(f"<td style=\"width: 8px; height: 8px; background-color: {color}\">&nbsp;</td>")
    txt_f.write("</tr>")
    txt_f.write("</table>")
  img_f.close(); txt_f.close(); lbl_f.close()
  print("\nDone ")

convert("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte", "test.html", 10)